See the Jupyter Notebook for a quick tutorial on the functionality of the code.



# Using the Dockerfile

To run the dockerfile, we must give some permission access to docker by running

```bash
xhost +"local:docker@"
```

then one can run

```bash
bash docker_run.sh
```

at the root.

If you want to start a jupyter notebook in there,

```bash
jupyter notebook --allow-root --port=8888 --ip=0.0.0.0 &
```

You can also run

```bash
python -c "import EvSim; EvSim.SSsky(106)"
```

to see the plot GUI after going through the jupyter notebook.
