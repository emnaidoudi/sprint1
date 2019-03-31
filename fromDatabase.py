from pymongo import MongoClient


client = MongoClient('mongodb://localhost:27017/')  

db = client.chatbot

collection = db.groups

def get_manager(group,year='2019'):
    print(year)
    return list(collection.find({"name":group,"year":year}))[0]["manager"]




