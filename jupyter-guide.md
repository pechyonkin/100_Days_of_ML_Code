# Useful Shell Commands

## Remote Access to IPython Notebooks via SSH [from [here](https://coderwall.com/p/ohk6cg/remote-access-to-ipython-notebooks-via-ssh)]

1) On the server: 

```remote_user@server$ ipython notebook --no-browser --port=8889```

2) On local machine, preferably in tmux:

```local_user@laptop$ ssh -N -f -L localhost:8891:localhost:8889 remote_user@server```

3) In browser:

```http://localhost:8891/tree?``` 



