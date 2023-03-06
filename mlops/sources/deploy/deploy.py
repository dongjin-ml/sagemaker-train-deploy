import os
import sys
import argparse
import subprocess
from distutils.dir_util import copy_tree

def install_packages(args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", os.path.join(args.prefix_deploy, "input/requirements/requirements.txt")])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "sagemaker"])
    
class deploy():
    
    def __init__(self, args):
         
        self.args = args
        print (self.args)
        
    def _create_endpoint(self,):
        
        from sagemaker.pytorch.model import PyTorchModel
        from sagemaker.serializers import JSONSerializer
        from sagemaker.deserializers import JSONDeserializer 
        print (f"Endpoint-name: {self.args.endpoint_name}")  
        
        
        print ("inference-code", os.listdir(os.path.join(self.args.prefix_deploy, "input", "inference-code")))
        print ("requirements", os.listdir(os.path.join(self.args.prefix_deploy, "input", "requirements")))
        
        print ("source_dir", os.path.join(self.args.prefix_deploy, "input", "inference-code"))
        print ("self.args.endpoint_name", self.args.endpoint_name)
        print ("self.args.instance_type", self.args.instance_type)
        print ("int(self.args.initial_instance_count)", int(self.args.initial_instance_count))
        print ("image_uri", self.args.image_uri)
        
        cloud_esimator = PyTorchModel(
            source_dir=os.path.join(self.args.prefix_deploy, "input", "inference-code"),
            entry_point="inference.py",
            framework_version=self.args.framework_version,
            py_version=self.args.py_version,
            image_uri=None, 
            model_data=self.args.model_data,
            role=self.args.execution_role,
            model_server_workers=int(self.args.model_server_workers)
        )
        
        cloud_predictor = cloud_esimator.deploy(
            endpoint_name=self.args.endpoint_name,
            instance_type=self.args.instance_type, 
            initial_instance_count=int(self.args.initial_instance_count),
            serializer=JSONSerializer('application/json'),
            deserializer=JSONDeserializer('application/json'),
            #wait=True,
            log=True,
        )
        
    def execution(self, ):
        
        self._create_endpoint() 
        print ("inference-code", os.listdir(os.path.join(self.args.prefix_deploy, "input", "inference-code")))
        print ("requirements", os.listdir(os.path.join(self.args.prefix_deploy, "input", "requirements")))
        
if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix_deploy", type=str, default="/opt/ml/processing/")
    parser.add_argument("--region", type=str, default="ap-northeast-2")
    parser.add_argument("--model_server_workers", type=str, default=1)
    parser.add_argument("--instance_type", type=str, default="ml.g4dn.xlarge")
    parser.add_argument("--initial_instance_count", type=str, default=1)
    
    parser.add_argument("--framework_version", type=str, default="framework_version")
    parser.add_argument("--py_version", type=str, default="py_version")
    parser.add_argument("--image_uri", type=str, default="image_uri")
    
    parser.add_argument("--model_data", type=str, default="model_data")
    parser.add_argument("--model_name", type=str, default="model_name")
    parser.add_argument("--endpoint_name", type=str, default="endpoint_name")
    parser.add_argument("--execution_role", type=str, default="execution_role")
    
    args, _ = parser.parse_known_args()
           
    print("Received arguments {}".format(args))
    os.environ['AWS_DEFAULT_REGION'] = args.region
    
    install_packages(args)
    
    dep = deploy(args)
    dep.execution()