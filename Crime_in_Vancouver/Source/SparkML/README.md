# Running Instructions
Running the program will either build a local file with prediction values for Crime Neighburhood in the future or upload it to the MongoDB database as the "Predictions" table. The option can be selected as per the "switch" key in the companion Json file (interpreted as Python3 dictionary.) The table can then be fed to the Tableau connector via commandline.

```
spark-submit --driver-memory 8G --executor-memory 8G --num-executors 4 --executor-cores 2 main.py
```
