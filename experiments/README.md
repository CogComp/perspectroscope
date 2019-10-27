# CLI for running experiments 

First install `amti`: 

```bash 
> pip install git+https://github.com/allenai/amti#egg=amti 
```


The main entry-point for experiments is the `setup_experiment.sh`: 

```bash 
> sh setup_experiment.sh  NUM_HITS  UNIQUE_ID  
```

 - `NUM_HITS`: an integer, to indicate the number of the HITS 
 - `UNIQUE_ID`: a semi-informative name to describe your expeirment. This name will be used in saving the logs for your experiment.  

Here is an example output you might see: 

```
> sh setup_experiment.sh   2 try1    
Number of the HITs:  2
Pilot Id:  try1
mkdir: step1_question_generation/pilot-try1: File exists
2019-10-17 21:32:26,244:INFO:amti.actions.create:Writing batch.
2019-10-17 21:32:26,259:INFO:amti.actions.create:Uploading batch to MTurk.
2019-10-17 21:32:30,733:INFO:amti.actions.create:Created 10 HITs.
2019-10-17 21:32:30,733:INFO:amti.actions.create:HIT Creation Complete.
2019-10-17 21:32:30,733:INFO:amti.clis.create:Finished.

    Preview HITs: https://workersandbox.mturk.com/

Will expire batch step1_question_generation/pilot-try1. Continue? [Ctrl+C to kill]

Expiring batch step1_question_generation/pilot-try1
 
```


Note that before running this experiment you should set the `AWS_PROFILE` enviromental variable (e.g., `export AWS_PROFILE=Aristo-mturk`) 
This would guide the CLI to look  for the right MTurk credentials stored in `~/.aws/credentials` file.

