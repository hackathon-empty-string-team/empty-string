# Source of truth

## welcome and initial setup
hi all, 
i think when we start with the EDA, it suffices if anyone just uses what they are used to (f.e. conda or whatever). However, afterwards, i think it could be helpful that everyone, always has exactly the same environment, same package/python versions, which is why i propose working with docker to minimize headaches and "but it works on my machine" issues. I think with this minimal setup below, we can fully focus on hacking while not having pain with painful stuff.

please feel free to add / change / challenge things!

- [notion page](https://glamorous-shawl-578.notion.site/Bird-Chirp-Classification-d7b3f86b0c114188b2782bd9b3d78c35)
- read how to setup docker for linux (ubuntu) [here](./docs/docker-setup.md)
- read how to setup docker for mac [here](https://docs.docker.com/desktop/install/mac-install/)

### how docker compose works
essentially, you just have to build the container with the services you want. if you're interested in it i can go into more detail just let me know. 

1. navigate to dc/dev and run:

```
docker compose up -d --build 
```

only use the `--build` flag the first time around, or if you want to rebuild the container (e.g. when having added a package you need in the container). **NOTE:** the `-d` flag stands for `detach` which means that your docker container runs in the background and does not log everything into your console.

2. then, to check whether everything worked hit: 

```
docker ps
```

3. for this specific setup, you can head to `localhost:8888` where jupyterlab is running. 

4. to create a new file (using jupytext, see below), just create a new .ipynb file, the .py file will be created automatically. all the changes you make in the notebook, will be reflected in the .py files which you then can use for your commits. 

now you shoold see the running docker containers. 

### what about huggingface spaces: 
- [here](./docs/huggingface-spaces.md), you can see what huggingface spaces is and how we can complement our github repo with it (credits to chat-gpt)

### jupytext - nice versioning of jupyter notebooks
since we are likely be working with jupyter notebooks alot, lets use jupytext. It automatically maps .ipynb to .py files with some magic. The .ipynb are in the gitignore, so we only have .py files nicely versioned in the repo. read more about it [here](https://jupytext.readthedocs.io/en/latest/)

### trunk based development
lets stick to trunk based. if you dont know what it is, read all about it [here](https://trunkbaseddevelopment.com/)

key take aways:

#### Trunk-Based Development: Key Points

1. **Single Main Branch**: All developers commit to the trunk or main branch.
2. **Short-Lived Branches**: Branches, if used, are short-lived and quickly merged back.
3. **Frequent Integrations**: Code changes are integrated frequently, often multiple times a day.
4.  **Feature Flags**: Incomplete features are managed with feature flags to maintain trunk stability.

#### Benefits

- **Reduced Integration Problems**: Early conflict detection and resolution.
- **Higher Code Quality**: Continuous testing ensures stable and high-quality code.
- **Simpler Workflow**: Less overhead managing branches and merges.
- **Enhanced Collaboration**: Encourages teamwork and code reviews.

#### Challenges

- **Discipline Required**: Developers must write clean, well-tested code.
- **Handling Incomplete Features**: Requires careful use of feature flags.

#### Best Practices

- **Frequent Commits**: Small, incremental changes reduce integration risks.
- **Comprehensive Testing**: Automated tests for codebase coverage.
- **Feature Flags**: Manage incomplete or experimental features.
- **Code Reviews**: Maintain quality and knowledge sharing.




