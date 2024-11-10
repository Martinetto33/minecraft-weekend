# A Minecraft clone with parallel procedural generation

## Disclaimer
The vast majority of this code was created by jdah. I was simply left astonished by his work and felt inspired to add something to it. Even this beautiful picture below is only his merit, not mine :)

![screenshot](screenshots/1.png)

## This is my thesis!
___
This project is my thesis for the Computer Science and Engineering bachelor's degree at the Alma Mater Studiorum - University of Bologna, Italy.
The aim was to use a NVIDIA GPU with CUDA enabled to parallelise the procedural generation of the height maps and blocks in the chunks of the world.
All the code I added is available in the <span style="color:cyan"><i>src/cuda</i></span> folder, with minor changes made to <span style="color:cyan"><i>src/world/gen/worldgen.c</i></span>.

The parallelised code proved to be faster than the serial code on my machine. While being efficient, this version can be still improved in many ways.

If you're bold enough to collaborate or to propose your changes, I'm definitely interested. :D

## Setting the project up
___
#### Requirements

- CUDA enabled GPU
- CUDA Toolkit installed and environment correctly set up; it's not that easy, it may take a while...
- Unix-like environment

You will probably need to know the compute capability of your GPU and change the `Makefile` in the root directory accordingly. Line 60 is what you're interested in, and it looks like this:

```aiignore
$(CUDA_COMPILER) -arch=sm_75 -dlink $(CUDA_OBJ) -o gpuCode.o -L/opt/cuda/nvvm/lib64 -lcudart
```
Change the `-arch=sm_xx` option with your compute capability without a dot written in place of `xx`. You can find your GPU compute capability by looking it up here: https://developer.nvidia.com/cuda-gpus. As an alternative, you can try completely removing the `-arch=sm_xx` option.

#### Building

You will need to install a bunch of libraries. Depending on your linux distribution, you may need to change the package manager used for installation. Here's a general command taken from https://github.com/jdah/minecraft-weekend/issues/40:
```
sudo apt install clang cmake libxss-dev libxxf86vm-dev libxkbfile-dev libxv-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev -y
```

##### Unix-like
`$ git clone --recurse-submodules https://github.com/Martinetto33/minecraft-weekend.git`\
`$ make`

The following static libraries under `lib/` must be built before the main project can be built:

- GLAD `lib/glad/src/glad.o`
- CGLM `lib/cglm/.libs/libcglm.a`
- GLFW `lib/glfw/src/libglfw3.a`
- libnoise `lib/noise/libnoise.a`

All of the above have their own Makefile under their respective subdirectory and can be built with `$ make libs`.
If libraries are not found, ensure that submodules have been cloned.

The game binary, once built with `$ make`, can be found in `./bin/`.

*Be sure* to run with `$ ./bin/game` out of the root directory of the repository.
If you are getting "cannot open file" errors (such as "cannot find ./res/shaders/*.vs"), this is the issue. 

##### Windows

Good luck 🤷‍♂️ probably try building under WSL and using an X environment to pass graphics through.

##### Troubleshooting

In case the project does not compile, throwing errors regarding implicit function declaration, modify the CMakeLists.txt files in the library folders as specified in this issue: https://github.com/jdah/minecraft-weekend/issues/97.

In summary, you need to go to **./Makefile** and modify it by adding the following line to the CFLAGS:

```
CFLAGS += -Wno-error=implicit-function-declaration
```

Then you need to go to **./lib/cglm/CMakeLists.txt** and look around line 36, right under some code performing ``string(REGEX REPLACE "/RTC(su|[1su])" "" ${flag_var} "${${flag_var}}")``. There, you will find a line like this:

```
add_compile_options(-Wall -Werror -O3)
```
You need to modify it as follows:
```
add_compile_options(-Wall -Werror -Wno-error=array-bounds -Wno-error=stringop-overflow -Wno-error=array-parameter -O3)
```

Now you should be able to run the make command in the root directory.
