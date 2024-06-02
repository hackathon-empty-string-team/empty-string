## welcome and minimal setup

In the following you find a minimal set of instructions on how to use this repo. 

- read how to setup docker for linux (ubuntu) [here](./docs/docker-setup.md)
- read how to setup docker for mac [here](https://docs.docker.com/desktop/install/mac-install/)

### how to use this repo

1. clone the repo

2. navigate to dc/dev and run:

```
docker compose up -d --build
```

only use the `--build` flag the first time around, or if you want to rebuild the container (e.g. when having added a package you need in the container). **NOTE:** the `-d` flag stands for `detach` which means that your docker container runs in the background and does not log everything into your console.

3. then, to check whether everything worked hit:

```
docker ps
```

4. for this specific setup, you can head to `localhost:8888` where jupyterlab is running.

5. to create a new file (using jupytext, see below), just create a new .ipynb file, the .py file will be created automatically. all the changes you make in the notebook, will be reflected in the .py files which you then can use for your commits.

now you should see the running docker containers.


### jupytext - nice versioning of jupyter notebooks

- we use [jupytext](https://jupytext.readthedocs.io/en/latest/)
- It automatically maps .ipynb to .py files with some magic
- The .ipynb are in the gitignore, so we only have .py files nicely versioned in the repo

### trunk based development

- lets stick to [trunk based development](https://trunkbaseddevelopment.com/)

### code format

- lets stick to [Black](https://black.readthedocs.io/en/stable/) for python and [Prettier](https://prettier.io/) for .md and other formats
- using docker for the purpose of formatting is really easy

  1.  `chmod +x format` so that the `format` file is executable
  2.  then simply use `./format` before adding your changes and all the files will be autoformatted
