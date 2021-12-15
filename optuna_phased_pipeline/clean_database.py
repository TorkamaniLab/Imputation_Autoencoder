import sys
import mysql.connector

print(sys.argv)
if(len(sys.argv)!=3):
    print("Usage: script.py <user@host> <database_name>")
    sys.exit()
user_name, host_name = sys.argv[1].split('@')
mydb = mysql.connector.connect(host=host_name,user=user_name)
mycursor = mydb.cursor()
study_name = sys.argv[2].replace('-','_')
#print("user:", user_name, "host:", host_name, "database:", study_name)
mycursor.execute("DROP DATABASE IF EXISTS " + study_name)
mycursor.execute("CREATE DATABASE IF NOT EXISTS " + study_name)
print("Cleaned mySQL database for", study_name)
