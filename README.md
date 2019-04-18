# PerspectroScope 

A Django-based web demo system for discovering diverse perspectives to claims.

## Setting up the system 
#### Getting started 

To install the python dependencies, simply do: 
```python 
pip3.6 install -r requirements.txt
```

- Check if Django is installed:
 ```
 $ python -m django --version
 ```

- Setup the database
```
> python3 manage.py makemigrations app  # you should see the migrations under "app/migrations"
> python3 manage.py sqlmigrate app 0001
> python3 manage.py migrate
```

- Run the app:
```
$ python3 manage.py runserver
```

## Details

The system has two layers: 

### Information Retrieval
The system using a retrieval engine in order to speed up the system.
To begin with, download an instance of a retrieval engine from https://elastic.co .
And index the two pools of perspective and evidence (included in [the perspectrum repository](https://github.com/CogComp/perspectrum/tree/master/data)).
The two indices have to be named `"perspective_pool_v0.2"` and  `"evidence_pool_v0.2"`, respectively.
[Here](README_elastic.md) is a brief instruction on how to use elasticsearch and how to index json files.


### Learned models 

Download the trained model files from [this link](https://drive.google.com/drive/folders/1B0XAWxn7xOsn1bRYCbZcSzh2HiABkx6p?usp=sharing)
and put them under `data/model/` (for instance, `data/model/relevance/perspectrum_relevance_lr2e-05_bs32_epoch-0.pth`).


## Citation 
If you find this useful, please consider citing the following work(s):
```
@inproceedings{chen2018perspectives,
  TODO
}
```
