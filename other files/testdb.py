import pandas as pd
import pymysql

mydb = pymysql.connect(
    host="localhost",
    user="jenguehard",
    password="Testsimplon_99",
    database="youtube"
    )

mycursor = mydb.cursor()

youtube_link = "https://www.youtube.com/watch?v=52UZJJ79Hrg"

query_vid = """SELECT * from videos WHERE url=%s"""
arg_vid = (youtube_link,)
mycursor.execute(query_vid, arg_vid)

result_vid = mycursor.fetchall()
print(result_vid)
video_id = result_vid[0][0]

query = """SELECT * from comments WHERE video_id = %s"""
arg = (video_id,)
mycursor.execute(query, arg)
result = mycursor.fetchall()
print(result)