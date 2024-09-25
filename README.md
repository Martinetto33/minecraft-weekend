# Minecraft, but I made it in 48 hours*

\* I've actually updated it since - [see this commit for the 48 hour version](https://github.com/jdah/minecraft-weekend/tree/cb19738305804b5734faa7118c1c784f26ff9463).

![screenshot](screenshots/1.png)

#### Features:
- Infinite, procedurally generated world
- Infinite height/depth
- Day/night cycle
- Biomes
- ECS-driven player and entities with full collision and movement
- Full RGB lighting
- Full transparency + translucency support
- Sprite blocks (flowers)
- Animated blocks (water + lava)
- Distance fog
- A whole lot of different block types
- More

#### Building

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

good luck 🤷‍♂️ probably try building under WSL and using an X environment to pass graphics through.

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
