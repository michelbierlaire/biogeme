from cysetuptools import setup
import platform

if platform.system() == 'Darwin':
    os.environ["CC"] = 'clang++'
    os.environ["CXX"] = 'clang++'
else:
    os.environ["CC"] = 'g++'
    os.environ["CXX"] = 'g++'


setup()


