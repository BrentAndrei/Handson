from roboflow import Roboflow

rf = Roboflow(api_key="Hi7KzCtDKQbRhd73y6Bf")
project = rf.workspace("research-xbc14").project("filipino-sign-language-recognition")
dataset = project.version(2).download("coco", location="./roboflow_fsl_phrases")
print("Downloaded to:", dataset.location)
