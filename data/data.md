If you want to save local data and push it to git, put under `data`.

If you want to have data in the repo but not track it on git (because of its size), you can make a new directory called `large_files` under `data`. The `.gitignore` will ignore any thing inside `data\large_files`