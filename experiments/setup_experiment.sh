#!/bin/bash
# make sure to set your AWS_PROFILE enviromental variable before running this!
# this would guide the cli to look  for your MTurk credentials stored in ~/.aws/credentials file.


URL=https://perspectroscope.com/annotator/withWiki/random_claim=true/

if [ -z "$1" ]; then
  echo "The number of the HITs were not specified (expected an integer)"
  exit 125
fi

if [ -z "$2" ]; then
  echo "No pilot-id argument was supplied (expected an integer or a description)"
  exit 125
fi

echo "Number of the HITs: " $1
echo "Pilot Id: " $2

# experiment type
EXPERIMENT_DIR="logs/pilot-$2"
mkdir $EXPERIMENT_DIR
cp -r definition/* $EXPERIMENT_DIR

python3 experiment_helpers.py --base ${URL} --size $1 >"$EXPERIMENT_DIR/data.jsonl"
python3 experiment_helpers.py --base ${URL} --size $1 --sandbox >"$EXPERIMENT_DIR/data-sandbox.jsonl"

#amti create-batch $EXPERIMENT_DIR ${EXPERIMENT_DIR}/data-sandbox.jsonl .

#echo "Will expire batch $EXPERIMENT_DIR. Continue? [Ctrl+C to kill]"
#read
#echo "Expiring batch $EXPERIMENT_DIR"
#amti expire-batch $EXPERIMENT_DIR

## Warning: the commands below will publish batches to the live system

# last_batch_dir=`ls -trd batch-* |tail -n1`

# echo
# echo "Will delete dir $last_batch_dir. Continue? [Ctrl+C to kill]"
# read
# echo "Deleting sandbox dir $last_batch_dir"
# rm -rf ${last_batch_dir}

echo
echo "Will submit live. Continue? [Ctrl+C to kill]"
read
cp -r live_definition/* $EXPERIMENT_DIR

SAVE_DIR=${EXPERIMENT_DIR}/out/
mkdir $SAVE_DIR

amti create-batch --live $EXPERIMENT_DIR ${EXPERIMENT_DIR}/data.jsonl ${EXPERIMENT_DIR}/out/
echo "Created live batch in directory: $EXPERIMENT_DIR"
