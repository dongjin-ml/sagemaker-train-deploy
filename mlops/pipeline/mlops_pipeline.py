import os
import time
import boto3
import shutil
import argparse
import sagemaker
import subprocess
from os import path
from datetime import datetime

from utils.s3 import s3_handler
from utils.ssm import parameter_store
from config.config import config_handler

from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role
from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.functions import Join
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.pytorch.processing import PyTorchProcessor
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.model_metrics import MetricsSource, ModelMetrics 
from sagemaker.workflow.steps import CacheConfig, ProcessingStep, TrainingStep
from sagemaker.workflow.lambda_step import LambdaStep, LambdaOutput, LambdaOutputTypeEnum
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.retry import StepRetryPolicy, StepExceptionTypeEnum, SageMakerJobExceptionTypeEnum, SageMakerJobStepRetryPolicy
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo, ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.fail_step import FailStep

class pipeline_tr():
    
    def __init__(self, args):
        
        self.args = args
        
        self.region = self.args.config.get_value("COMMON", "region")
        self.s3 = s3_handler()
        self.pm = parameter_store(self.region)
        self._env_setting()        
        
    def _env_setting(self, ):
        
        self.role = get_execution_role()
        
        self.prefix = self.args.config.get_value("COMMON", "prefix")
        self.model_name = self.args.config.get_value("COMMON", "model_name")
        self.pipeline_name = self.prefix + self.args.config.get_value("PIPELINE", "name") + "-" + self.model_name
        self.pipeline_session = PipelineSession()
        self.cache_config = CacheConfig(
            enable_caching=self.args.config.get_value("PIPELINE", "enable_caching", dtype="boolean"),
            expire_after=self.args.config.get_value("PIPELINE", "expire_after")
        )    
        
        self.pm.put_params(key="PREFIX", value=self.prefix, overwrite=True)
        self.pm.put_params(key="".join([self.prefix, "REGION"]), value=self.region, overwrite=True)
        self.pm.put_params(key="".join([self.prefix, "BUCKET"]), value=sagemaker.Session().default_bucket(), overwrite=True)
        self.pm.put_params(key="".join([self.prefix, "SAGEMAKER-ROLE-ARN"]), value=self.role, overwrite=True)
        self.pm.put_params(key="".join([self.prefix, "ACCOUNT-ID"]), value=boto3.client("sts").get_caller_identity().get("Account"), 
                           overwrite=True)
        self.pm.put_params(key=self.prefix + "PIPELINE-NAME", value=self.pipeline_name, overwrite=True)
        
        print (f" == Envrionment parameters == ")
        print (f"   SAGEMAKER-ROLE-ARN: {self.role}")
        print (f"   PREFIX: {self.prefix}")
        print (f"   BUCKET: {sagemaker.Session().default_bucket()}")
        print (f"   ACCOUNT-ID: {boto3.client('sts').get_caller_identity().get('Account')}")
        
    def _step_preprocessing(self, ):
        
        def _download_data():
            
            ## download and move data
            strTmpDir = "./dataset_tmp"
            strInputDir = "cifar-10-batches-py"
            os.makedirs(strTmpDir, exist_ok=True)
            subprocess.check_call(["wget", "-O", os.path.join(strTmpDir, "cifar-10-python.tar.gz"), "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",])
            subprocess.check_call(["tar", "xfvz", os.path.join(strTmpDir, "cifar-10-python.tar.gz"), "-C", strTmpDir, ])
            os.makedirs(os.path.join(strTmpDir, "data"), exist_ok=True)
            subprocess.check_call(["mv", os.path.join(strTmpDir, strInputDir), os.path.join(strTmpDir, "data"),])
            subprocess.check_call(["rm", os.path.join(strTmpDir, "cifar-10-python.tar.gz"),])
            
            ## upload to S3 and regiter path to parameter store
            source_dir, target_bucket, target_dir = os.path.join(strTmpDir, "data"), self.pm.get_params(key=self.prefix+"BUCKET"), self.prefix+"data"
            self.s3.upload_dir(source_dir, target_bucket, target_dir)
            self.pm.put_params(key="".join([self.prefix, "DATA-PATH"]), value=f"s3://{target_bucket}/{target_dir}", overwrite=True)
            
            ## clean up
            if os.path.isdir(strTmpDir): shutil.rmtree(strTmpDir)
            
            return strInputDir

        strInputDir = _download_data()            
        #strInputDir = "cifar-10-batches-py"
        
        sklearn_processor = SKLearnProcessor(
            framework_version=self.args.config.get_value("PREPROCESSING", "framework_version"), # https://docs.aws.amazon.com/sagemaker/latest/dg/sklearn.html, 0.20.0, 0.23-1, 1.0-1.
            role=self.role,
            #instance_type="local",
            instance_type=self.args.config.get_value("PREPROCESSING", "instance_type"),
            instance_count=self.args.config.get_value("PREPROCESSING", "instance_count", dtype="int"),
            base_job_name="preprocessing", # bucket에 보이는 이름 (pipeline으로 묶으면 pipeline에서 정의한 이름으로 bucket에 보임)
            sagemaker_session=self.pipeline_session ## Add
        )
        
        prefix_prep = "/opt/ml/processing/"
        step_args = sklearn_processor.run(
            code='../sources/preprocessing/preprocessing.py',
            inputs=[
                ProcessingInput(
                    input_name="input",
                    source=self.pm.get_params(key=self.prefix + "DATA-PATH"),
                    destination=os.path.join(prefix_prep, "input")
                ),
                ProcessingInput(
                    input_name="requirements",
                    source='../sources/preprocessing/requirements.txt',
                    destination=os.path.join(prefix_prep, "input", "requirements")
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="train_data",
                    source=os.path.join(prefix_prep, "output", "train"),
                    destination=Join(
                        on="/",
                        values=[
                            "s3://{}".format(self.pm.get_params(key=self.prefix + "BUCKET")),
                            self.pipeline_name,
                            ExecutionVariables.PROCESSING_JOB_NAME,
                            "train_data",
                        ],
                    ),
                ),
                ProcessingOutput(
                    output_name="validation_data",
                    source=os.path.join(prefix_prep, "output", "validation"),
                    destination=Join(
                        on="/",
                        values=[
                            "s3://{}".format(self.pm.get_params(key=self.prefix + "BUCKET")),
                            self.pipeline_name,
                            ExecutionVariables.PROCESSING_JOB_NAME,
                            "validation_data",
                        ],
                    ),
                ),
                ProcessingOutput(
                    output_name="test_data",
                    source=os.path.join(prefix_prep, "output", "test"),
                    destination=Join(
                        on="/",
                        values=[
                            "s3://{}".format(self.pm.get_params(key=self.prefix + "BUCKET")),
                            self.pipeline_name,
                            ExecutionVariables.PROCESSING_JOB_NAME,
                            "test_data",
                        ],
                    ),
                )
            ],
            arguments=["--prefix_prep", prefix_prep, "--input_dir", strInputDir, "--region", self.region],
            job_name="preprocessing",
        )
        
        self.preprocessing_process = ProcessingStep(
            name="PreprocessingProcess", ## Processing job이름
            step_args=step_args,
            cache_config=self.cache_config,
        )
        
        print ("  \n== Preprocessing Step ==")
        print ("   \nArgs: ", self.preprocessing_process.arguments.items())

    def _step_training(self, ):
        
        self.estimator = PyTorch(
            source_dir="../sources/train",
            entry_point="cifar10.py",
            role=self.role,
            framework_version=self.args.config.get_value("TRAINING", "framework_version"),
            py_version=self.args.config.get_value("TRAINING", "py_version"),
            instance_type=self.args.config.get_value("TRAINING", "instance_type"),
            #instance_type="local_gpu",
            instance_count=self.args.config.get_value("TRAINING", "instance_count", dtype="int"),
            sagemaker_session = self.pipeline_session,
            output_path=Join(
                on="/",
                values=[
                    "s3://{}".format(self.pm.get_params(key=self.prefix + "BUCKET")),
                    self.pipeline_name,
                ],
            ),
        )
        
        step_training_args = self.estimator.fit(
            job_name="training",
            inputs={
                "TR": self.preprocessing_process.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri,
                "VAL": self.preprocessing_process.properties.ProcessingOutputConfig.Outputs["validation_data"].S3Output.S3Uri,
                "TE": self.preprocessing_process.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri,
            },
            logs="All",
        )
          
        self.training_process = TrainingStep(
            name="TrainingProcess",
            step_args=step_training_args,
            cache_config=self.cache_config,
            depends_on=[self.preprocessing_process],
            retry_policies=[                
                # retry when resource limit quota gets exceeded
                SageMakerJobStepRetryPolicy(
                    exception_types=[SageMakerJobExceptionTypeEnum.RESOURCE_LIMIT],
                    expire_after_mins=180,
                    interval_seconds=600,
                    backoff_rate=1.0
                ),
            ]
        )
        
        print ("  \n== Training Step ==")
        print ("   \nArgs: ", self.training_process.arguments.items())
                
    def _step_evaluation(self, ):
        
        #https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job-frameworks-pytorch.html
        
        #Initialize the PyTorchProcessor
        eval_processor = PyTorchProcessor(
            image_uri = self.training_process.properties.AlgorithmSpecification.TrainingImage, 
            framework_version=None, #self.args.config.get_value("EVALUATION", "framework_version"),
            py_version=None, #self.args.config.get_value("EVALUATION", "py_version"),
            role=self.role,
            instance_type=self.args.config.get_value("EVALUATION", "instance_type"),
            instance_count=self.args.config.get_value("EVALUATION", "instance_count", dtype="int"),
            base_job_name="evaluation", # bucket에 보이는 이름 (pipeline으로 묶으면 pipeline에서 정의한 이름으로 bucket에 보임)
            sagemaker_session=self.pipeline_session,
        )
        
        self.evaluation_report = PropertyFile(
            name="EvaluationReport",
            output_name="evaluation-metrics",
            path="evaluation-" + self.model_name +  ".json",
        )
        
        prefix_eval = "/opt/ml/processing/"
        step_args = eval_processor.run(
            job_name="evaluation", # Evaluation job name. If not specified, the processor generates a default job name, based on the base job name and current timestamp.
                                   # 이걸 넣어야 캐시가 작동함, 안그러면 프로세서의 base_job_name 이름뒤에 날짜 시간이 붙어서 캐시 동작 안함
            code='./evaluation/evaluation.py', #소스 디렉토리 안에서 파일 path
            source_dir="../sources/", #현재 파일에서 소스 디렉토리 상대경로 # add script.py and requirements.txt here
            
            inputs=[
                ProcessingInput(
                    source=self.training_process.properties.ModelArtifacts.S3ModelArtifacts,
                    destination=os.path.join(prefix_eval, "model") #"/opt/ml/processing/model"
                ),
                ProcessingInput(
                    source=self.preprocessing_process.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri,
                    destination=os.path.join(prefix_eval, "test") #"/opt/ml/processing/test"
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation-metrics",
                    source=os.path.join(prefix_eval, "evaluation"), #"/opt/ml/processing/evaluation",
                    destination=Join(
                        on="/",
                        values=[
                            "s3://{}".format(self.pm.get_params(key=self.prefix + "BUCKET")),
                            self.pipeline_name,
                            ExecutionVariables.PROCESSING_JOB_NAME,
                            "evaluation-metrics",
                        ],
                    ),
                )
            ],
            arguments=["--s3_model_path", self.training_process.properties.ModelArtifacts.S3ModelArtifacts, \
                       "--training_image_uri", self.training_process.properties.AlgorithmSpecification.TrainingImage, \
                       "--region", self.region, "--model_name", self.model_name, \
                       "--prefix_eval", prefix_eval]
        )
        
        self.evaluation_process = ProcessingStep(
            name="EvaluationProcess", ## Processing job이름
            step_args=step_args,
            depends_on=[self.preprocessing_process, self.training_process],
            property_files=[self.evaluation_report],
            cache_config=self.cache_config,
            retry_policies=[                
                # retry when resource limit quota gets exceeded
                SageMakerJobStepRetryPolicy(
                    exception_types=[SageMakerJobExceptionTypeEnum.RESOURCE_LIMIT],
                    expire_after_mins=180,
                    interval_seconds=600,
                    backoff_rate=1.0
                ),
            ]
        )
        
        print ("  \n== Evaluation Step ==")
        print ("   \nArgs: ", self.evaluation_process.arguments.items())
    
    def _step_model_registration(self, ):
      
        self.model_package_group_name = ''.join([self.prefix, self.model_name])
        self.pm.put_params(key=self.prefix + "MODEL-GROUP-NAME", value=self.model_package_group_name, overwrite=True)
                                                                              
        model_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri=Join(
                    on="/",
                    values=[
                        self.evaluation_process.properties.ProcessingOutputConfig.Outputs["evaluation-metrics"].S3Output.S3Uri,
                        #print (self.evaluation_process.arguments.items())로 확인가능
                        f"evaluation-{self.model_name}.json"
                    ],
                ),
                content_type="application/json")
        )
                                       
        self.register_process = RegisterModel(
            name="ModelRegisterProcess", ## Processing job이름
            estimator=self.estimator,
            image_uri=self.training_process.properties.AlgorithmSpecification.TrainingImage, 
            model_data=self.training_process.properties.ModelArtifacts.S3ModelArtifacts,
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=self.args.config.get_value("MODEL_REGISTER", "inference_instances", dtype="list"),
            transform_instances=self.args.config.get_value("MODEL_REGISTER", "transform_instances", dtype="list"),
            model_package_group_name=self.model_package_group_name,
            approval_status=self.args.config.get_value("MODEL_REGISTER", "model_approval_status_default"),
            ## “Approved”, “Rejected”, or “PendingManualApproval” (default: “PendingManualApproval”).
            model_metrics=model_metrics,
            depends_on=[self.evaluation_process]
        )
        
        print ("  \n== Registration Step ==")
        
    def _step_condition(self, ):
        
        
        # https://docs.aws.amazon.com/ko_kr/sagemaker/latest/dg/build-and-manage-steps.html#step-type-condition
        # 조건문 종류: https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_model_building_pipeline.html#conditions
        
        self.condition_acc = ConditionGreaterThanOrEqualTo(
            left=JsonGet(
                step_name=self.evaluation_process.name,
                property_file=self.evaluation_report,
                json_path="performance_metrics.accuracy.value" ## evaluation.py에서 json으로 performance를 기록한 대로 한다. 
                                                               ## 즉, S3에 저장된 evaluation-<model_name>.json 파일안에 있는 값을 적어줘야 한다. 
            ),
            right=self.args.config.get_value("CONDITION", "thesh_acc", dtype="float"),
        )
        
        self.condition_prec = ConditionGreaterThanOrEqualTo(
            left=JsonGet(
                step_name=self.evaluation_process.name,
                property_file=self.evaluation_report,
                json_path="performance_metrics.prec.value" ## evaluation.py에서 json으로 performance를 기록한 대로 한다. 
                                                           ## 즉, S3에 저장된 evaluation-<model_name>.json 파일안에 있는 값을 적어줘야 한다. 
            ),
            right=self.args.config.get_value("CONDITION", "thesh_prec", dtype="float"),
        )
        
        self.condition_process = ConditionStep(
            name="CheckCondition",
            display_name="CheckCondition",
            conditions=[self.condition_acc, self.condition_prec], ## 여러 조건 함께 사용할 수 있음
            if_steps=[self.approval_process],
            else_steps=[self.fail_process],
            depends_on=[self.register_process]
        )
        
        print ("  \n== Condition Step ==")
        print ("   \nArgs: ", self.condition_process.arguments)
        
    def _step_approval(self, ):
                
        # Lambda helper class can be used to create the Lambda function
        ## <Infortant> "lambda role" is needed!!
        ## Refer to "2.3 Create role for lambda" in 0.setup.ipynb
        
        approval_lambda = Lambda(            
            function_name=''.join([self.args.config.get_value("COMMON", "prefix"), "LambdaApprovalStep"]),
            execution_role_arn=self.pm.get_params(key=self.prefix + "LAMBDA-ROLE-ARN"), 
            script="../sources/approval/approval_lambda_handler.py",
            handler="approval_lambda_handler.lambda_handler",
            session=self.pipeline_session,
            timeout=900 # default: 120s 
        )
          
        strCurTime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        cache_config = CacheConfig(enable_caching=False,)    
        self.approval_process = LambdaStep(
            name="LambdaModelApprovalProcess",
            description="Lambda for model approval",
            lambda_func=approval_lambda,
            inputs={
                "model_package_group_name": self.model_package_group_name,
                "region": self.region,
            },
            outputs=[
                LambdaOutput(output_name="statusCode", output_type=LambdaOutputTypeEnum.String),
            ],
            cache_config=cache_config,
            #depends_on=[self.condition_process]
        )
        
    def _step_fail(self, ):
        
        self.fail_process = FailStep(
            name="ConditionFail",
            error_message=Join(
                on=" ",
                values=["Execution failed due to performance threshold"]
            ),
        )
        
    def _step_deploy(self, ):
        
        sklearn_processor = SKLearnProcessor(
            framework_version=self.args.config.get_value("DEPLOY", "processing_framework_version"),
            # https://docs.aws.amazon.com/sagemaker/latest/dg/sklearn.html, 0.20.0, 0.23-1, 1.0-1.
            role=self.role,
            #instance_type="local",
            instance_type=self.args.config.get_value("DEPLOY", "processing_instance_type"),
            instance_count=self.args.config.get_value("DEPLOY", "processing_instance_count", dtype="int"),
            base_job_name="deploy", # bucket에 보이는 이름 (pipeline으로 묶으면 pipeline에서 정의한 이름으로 bucket에 보임)
            sagemaker_session=self.pipeline_session ## Add
        )
        
        prefix_deploy = "/opt/ml/processing/"
        endpoint_name = f"endpoint-{self.model_name}{int(time.time())}"
        step_args = sklearn_processor.run(
            code='../sources/deploy/deploy.py',
            inputs=[
                ProcessingInput(
                    input_name="inference-code",
                    source="../sources/deploy/inference.py",
                    destination=os.path.join(prefix_deploy, "input", "inference-code")
                ),
                ProcessingInput(
                    input_name="requirements",
                    source='../sources/deploy/requirements.txt',
                    destination=os.path.join(prefix_deploy, "input", "requirements")
                ),
            ],
            arguments=["--prefix_deploy", prefix_deploy, "--region", self.region, \
                       "--model_server_workers", str(self.args.config.get_value("DEPLOY", "model_server_workers", dtype="int")), \
                       "--instance_type", self.args.config.get_value("DEPLOY", "instance_type"), \
                       "--initial_instance_count", str(self.args.config.get_value("DEPLOY", "initial_instance_count", dtype="int")), \
                       "--image_uri", self.training_process.properties.AlgorithmSpecification.TrainingImage, \
                       "--model_data", self.training_process.properties.ModelArtifacts.S3ModelArtifacts, \
                       "--model_name", self.model_name, \
                       "--endpoint_name", endpoint_name, \
                       "--execution_role", self.role, \
                       "--framework_version", self.args.config.get_value("DEPLOY", "framework_version"), \
                       "--py_version", self.args.config.get_value("DEPLOY", "py_version")],
            job_name="deploy",
        )
        self.pm.put_params(key=self.prefix + "ENDPOINT-NAME", value=endpoint_name, overwrite=True)
        
        self.deploy_process = ProcessingStep(
            name="DeployProcess", ## Processing job이름
            step_args=step_args,
            depends_on=[self.approval_process],
            cache_config=self.cache_config,
        )
    
    def _get_pipeline(self, ):
        
        
        pipeline = Pipeline(name=self.pipeline_name,
                           steps=[self.preprocessing_process, self.training_process, self.evaluation_process, \
                                  self.register_process, self.condition_process, self.deploy_process],)

        return pipeline
                            
    def execution(self, ):
         
        self._step_preprocessing()
        self._step_training()
        self._step_evaluation()
        self._step_model_registration()
        self._step_approval()
        self._step_fail()
        self._step_condition()
        self._step_deploy()
        
        pipeline = self._get_pipeline()
        pipeline.upsert(role_arn=self.role) ## Submit the pipeline definition to the SageMaker Pipelines service 
        execution = pipeline.start()
        execution.describe()

if __name__ == "__main__":
    
    
    strBasePath, strCurrentDir = path.dirname(path.abspath(__file__)), os.getcwd()
    os.chdir(strBasePath)
    print ("==================")
    print (f"  Working Dir: {os.getcwd()}")
    print (f"  You should execute 'mlops_pipeline.py' in 'pipeline' directory'") 
    print ("==================")
    
    parser = argparse.ArgumentParser()
    #parser.add_argument("--mode", type=str, default="docker")
    args, _ = parser.parse_known_args()
    args.config = config_handler()
    
    print("Received arguments {}".format(args))
    #os.environ['AWS_DEFAULT_REGION'] = args.config.get_value("COMMON", "region")
    
    pipe_tr = pipeline_tr(args)
    pipe_tr.execution()