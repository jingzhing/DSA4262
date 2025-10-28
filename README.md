# DSA4262

How 2 Run With Docker... Check Ronin Instance Now. Dir will show data_in and data_out respectively. I have run 1 for task 1 dataset1 and task 2 SGNex_A549_directRNA_replicate5_run1.

To Do:
Run the remaining tasks1 and task2 files inside the same ronin instance
Docker Image of model is uploaded to this git under packages.

Once in utunbu for the instance: 
1) Pull the image into the cloud:          docker pull ghcr.io/jingzhing/m6a-hgb-ghost-infer:v2 (if its not there or if any updates)
2) Try to run the dataset from task1: by first copying into cloud (for your convinience):            scp -i ~/xxx.pem C:\Users\xxxx\xxxx\dataset2.json.gz ubuntu@xxxxx:~/data_in/
3) Run and produce output file            docker run --rm \-v ~/data_in:/data_in \-v ~/data_out:/data_out \ghcr.io/jingzhing/m6a-hgb-ghost-infer:v2 \predict --json /data_in/dataset2.json.gz \--model /opt/model/model_tuned.joblib \--output /data_out/preds2.csv
4) Pull all task 2 files into local (?) Idk if its a good idea but this code works, dont change anything just cd into data_in folder and run:                                                                                                                               while read run; do   echo "Downloading $run";   aws s3 cp --no-sign-request     s3://sg-nexdata/data/processed_data/m6Anet/${run}/data.json     ~/data_in/task2/${run}.json; done < runlist.txt
5) Run Predict on the json files that will be auto-labelled. Double check also that im using the correct file... lol it should be 
http://sg-nex-data.s3-website-ap-southeast-1.amazonaws.com/#data/processed_data/m6Anet/
code: docker run --rm \
  -v ~/data_in:/data_in \
  -v ~/data_out:/data_out \
  ghcr.io/jingzhing/m6a-hgb-ghost-infer:v2 \
  predict --json /data_in/task2/SGNex_A549_directRNA_replicate5_run1.json \
          --model /opt/model/model_tuned.joblib \
          --output /data_out/task2/SGNex_A549_directRNA_replicate5_run1_preds.csv
