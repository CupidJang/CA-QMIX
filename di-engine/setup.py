# Modified for CA-QMIX: packages ding and dizoo.smac2 only.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from setuptools import setup, find_packages

setup(
    name='DI-engine',
    version='0.5.3+caqmix',
    description='DI-engine v0.5.3 with CA-QMIX, QPLEX, Qatten and SMACv2 support',
    url='https://github.com/opendilab/DI-engine',
    license='Apache License, Version 2.0',
    packages=find_packages(include=('ding', 'ding.*', 'dizoo', 'dizoo.*')),
    package_data={
        package_name: ['*.yaml', '*.xml', '*cfg']
        for package_name in find_packages(include=('ding.*', ))
    },
    python_requires='>=3.7',
    install_requires=[
        'setuptools<=66.1.1',
        'yapf==0.29.0',
        'gym==0.25.1',
        'gymnasium',
        'torch>=1.1.0',
        'numpy>=1.18.0,<2',
        'DI-treetensor>=0.4.0',
        'DI-toolkit>=0.1.0',
        'trueskill',
        'tensorboardX>=2.2',
        'wandb<=0.19.0',
        'matplotlib',
        'easydict>=1.9',
        'pyyaml',
        'enum_tools',
        'cloudpickle',
        'hickle',
        'tabulate',
        'click>=7.0.0',
        'flask<=2.0.3',
        'werkzeug<=2.0.3',
        'requests',
        'responses',
        'URLObject',
        'pynng',
        'sniffio',
        'redis',
        'mpire>=2.3.5',
        'einops',
    ],
)
