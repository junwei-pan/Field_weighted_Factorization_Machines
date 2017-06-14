# One Week Dataset(20170524-20170530 for train, 20170531 for test)

Training dataset: 3,571,405 samples, 1,556,404 positive and 2,015,001 negative, positive sample has done 0.003 downsampleing

Testing dataset: 7,990,874 samples, 15,759 positive and 7,975,115 negative, total dataset has done 0.1 downsampling.

# Field and Features:

15 Fields.

Number of uniq features for each field, only those features occurs more than 10 times in the training data will be preserved.

|Fields|#Uniq Features|
|---|---|
|PUBLISHER_ID|41|
|PAGE_TLD|5,647|
|SUBDOMAIN|9,640|
|LAYOUT_ID|22|
|HOUR_OF_DAY|24|
|DAY_OF_WEEK|7|
|GENDER|3|
|AD_POSITION_ID|7|
|AGE_BUCKET|7|
|ADVERTISER_ID|554|
|AD_ID|6,605|
|CRRATIVE_ID|4,964|
|DEVICE_TYPE_ID|4|
|LINE_ID|1992|
|USER_ID|4,771|

# Performance

|Model|AUC|
|---|---|
|PNN1|0.848153|
|PNN2|0.854713|
|PNN1_Fixed|0.857946|
