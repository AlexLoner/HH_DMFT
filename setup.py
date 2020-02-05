import setuptools
import os
setuptools.setup(
     name='hh_dmft',  
     version='1.0',
     author= ["Lukianov Alex", "Neverov Viacheslav"],
     author_email=["lukyanov9567@gmail.com", "slavesta10@gmail.com"],
     description='this line in progress',
     #long_description=long_description,
     #long_description_content_type="text/markdown",
     url="https://github.com/AlexLoner/HH_DMFT",
    packages = ['hh_dmft'],
    package_data = {'' : ['test_data/*.txt']},
    #include_package_data=True,
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: Linux",
     ],
 )
