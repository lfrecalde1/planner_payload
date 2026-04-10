from setuptools import find_packages, setup

package_name = 'planner_payload'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fer',
    maintainer_email='lfrecalde1@espe.edu.ec',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'singlepayloadplanner = planner_payload.main_quadrotor_payload:main',
            'doublepayloadplanner = planner_payload.main_two_quadrotor_payload:main',
            'quadrotorpolicy = planner_payload.main_quadrotor:main',
            'single_new = planner_payload.main_quadrotor_payload_new:main',
            'single_jerk = planner_payload.main_quadrotor_payload_jerk:main',
            'inner_ipm_pendulum = planner_payload.inner_ipm_pendulum_node:main',
            'quadrotor_cable_dynamics = planner_payload.casadi_cable_dynamics_node:main',
        ],
    },
)
