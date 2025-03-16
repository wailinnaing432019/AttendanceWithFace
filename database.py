import mysql.connector

def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",  # Change if needed
        password="",  # Change if needed
        database="attendance_system"
    )
