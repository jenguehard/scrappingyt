# Project Scrapping Youtube

### Overview

This project allows you to get an analysis of the comments of a Youtube video.

Use python 3.6 version

In order to start the Streamlit application on local, please install the librairies inside the ```requirements.txt``` file.

Please create a Youtube API-Key with this [Google guide](https://developers.google.com/youtube/v3/getting-started) and put it into a ```config.py``` file that you need to create.

Launch the application on the terminal with the following command :

```streamlit run app.py```

Copy and paste your Youtube Link inside the dedicated file and press Enter. Shortly after the analysis of the comment will appear.

### MySQL set-up

This project requires a MySQL server to store the data and be able to retrieve it quickly based on the link of the video. As the analysis can take time this allows for quick analysis.

In this project, my MySQL server is in localhost but this information can be changed depending on your requirements.

The structure of the DB is the following : 

![dbstructure](https://github.com/jenguehard/scrappingyt/blob/master/images/Database%20structure.png)
