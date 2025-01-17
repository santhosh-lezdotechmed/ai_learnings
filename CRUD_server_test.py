import pymysql
import http
class ContactBook:
    def __init__(self):
        self.db_connection = self.connect_to_database()

    def connect_to_database(self):
        try:
            db_connection = pymysql.connect(
                host="localhost",
                user="root",
                password="",  
                database="contact_book",
                port=3306
            )
            print("Database connection successful!")
            return db_connection
        except pymysql.MySQLError as e:
            print(f"Error connecting to MySQL: {e}")
            return None

    def create_contact(self, name, phone_no):
        if self.db_connection:
            try:
                my_database = self.db_connection.cursor()
                sql_statement = "INSERT INTO contacts (name, phone_no) VALUES (%s, %s)"
                values = (name, phone_no)
                my_database.execute(sql_statement, values)
                self.db_connection.commit()
                print("Contact added successfully!")
            except pymysql.MySQLError as e:
                print(f"Error adding contact: {e}")

    def load_contacts(self):
        if self.db_connection:
            try:
                my_database = self.db_connection.cursor()
                sql_statement = "SELECT name, phone_no FROM contacts"  
                my_database.execute(sql_statement)
                contacts = my_database.fetchall()
                print(contacts)
                if contacts:
                    return  contacts

                else:
                   return []
            except pymysql.MySQLError as e:
                print(f"Error loading contacts: {e}")

        return []

    def update_contact(self, old_name, old_phone_no, new_name, new_phone_no):
        if self.db_connection:
            try:
                my_database = self.db_connection.cursor()
                sql_statement = "UPDATE contacts SET name = %s, phone_no = %s WHERE name = %s AND phone_no = %s"
                values = (new_name, new_phone_no, old_name, old_phone_no)
                my_database.execute(sql_statement, values)
                self.db_connection.commit()
                if my_database.rowcount > 0:
                    print("Contact updated successfully!")
                else:
                    print("No matching contact found to update.")
            except pymysql.MySQLError as e:
                print(f"Error updating contact: {e}")

    def delete_contact(self, name, phone_no):
        if self.db_connection:
            try:
                my_database = self.db_connection.cursor()
                sql_statement = "DELETE FROM contacts WHERE name = %s AND phone_no = %s"
                values = (name, phone_no)
                my_database.execute(sql_statement, values)
                self.db_connection.commit()
                if my_database.rowcount > 0:
                    print("Contact deleted successfully!")
                else:
                    print("No matching contact found to delete.")
            except pymysql.MySQLError as e:
                print(f"Error deleting contact: {e}")

    def menu(self):
        while True:
            print("\n--- Contact Book ---")
            print("C. Add Contact")
            print("R. View Contacts")
            print("E. Update Contact")
            print("D. Delete Contact")
            print("O. Exit")
            choice = input("Enter your choice: ")

            if choice == "C":
                name = input("Enter the contact name: ")
                phone_no = input("Enter the phone number: ")
                self.create_contact(name, phone_no)
            elif choice == "R":
                print("\nContact List:")
                self.load_contacts()  
            elif choice == "E":
                old_name = input("Enter the old contact name: ")
                old_phone_no = input("Enter the old phone number: ")
                new_name = input("Enter the new contact name: ")
                new_phone_no = input("Enter the new phone number: ")
                self.update_contact(old_name, old_phone_no, new_name, new_phone_no)
            elif choice == "D":
                name = input("Enter the contact name to delete: ")
                phone_no = input("Enter the phone number to delete: ")
                self.delete_contact(name, phone_no)
            elif choice == "O":
                print("Exiting Contact Book. Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")

contact_book = ContactBook()

# contact_book.menu()




import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from CRUD import ContactBook 

# Define the request handler
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler,ContactBook):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"message": "Welcome to the REST API!"}).encode("utf-8"))
        elif self.path == "/contacts":

            cont_f = ContactBook()
            cont= cont_f.load_contacts()
            print("cont")
            print(cont)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"message": "Welcome to the REST API!","data":cont}).encode("utf-8"))
        # elif self.path == "/hello":
        #     self.send_response(200)
        #     self.send_header("Content-Type", "application/json")
        #     self.end_headers()
        #     self.wfile.write(json.dumps({"greeting": "Hello, world!"}).encode("utf-8"))
        else:
            self.send_error(404, "Resource not found")

    def do_POST(self):
        if self.path == "/contacts":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data)
                print(data)
                cont_f = ContactBook()
                cont= cont_f.create_contact(data['name'],data['ph'])
                response = {"msg":"Contact"}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode("utf-8"))
            except json.JSONDecodeError:
                self.send_error(400, "Invalid JSON")
        else:
            self.send_error(404, "Resource not found")

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from routes import SimpleHTTPRequestHandler 
from CRUD import ContactBook 
# Run the server
def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler, port=8080):
    server_address = ("", port)
    httpd = server_class(server_address, handler_class)
    print(f"Server running on http://127.0.0.1:{port}")
    # ContactBook.connect_to_database
    httpd.serve_forever()

if __name__ == "__main__":
    run()


fix the errors in the files these are three files