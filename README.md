# Bigdata


# Instalar spark

```
Instalar o Java Runtime Environment
testar com cmd
java --version

Se n├úo tiver instalado correr:
conda install -c conda-forge openjdk
```

Vantagens de instalar o Java com o anaconda:
Environment Management: Installing Java through conda keeps it nicely contained within a conda environment. This helps prevent conflicts if you need different Java versions for different projects.


1. conda create --name bigdate
2. conda activate bigdata
3. conda install -c conda-forge pyspark
4. conda install -c conda-forge findspark
4. conda install -c conda-forge jupyterlab


Uma alternativa ao anaconda mais r├ípido:
````
conda install mamba -n base -c conda-forge
````

Depois funciona tudo normalmente como com o conda. Se quiserem instalar o mamba no ambiente
onde est├úo a trabalhar basta fazer:

````
conda install mamba
````

Se quiserem utilizar o code convem instalar o ````ipykernel````

````

mamba install -c conda-forge ipykernel

# se n├úo funcionar

conda install -c conda-forge ipykernel

````

