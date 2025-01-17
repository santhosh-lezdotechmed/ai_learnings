# Server Initialization

from http.server import HTTPServer
from routes import SimpleHTTPRequestHandler

def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler, port=8080):
    server_address = ("", port)
    httpd = server_class(server_address, handler_class)
    print(f"Server running on http://127.0.0.1:{port}")
    httpd.serve_forever()

if __name__ == "__main__":
    run()
    
    
# HTTP Request Handling

import json
from http.server import BaseHTTPRequestHandler
from CRUD import ContactBook

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    contact_book = ContactBook()

    def _set_headers(self, status_code=200):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"message": "Welcome to the Contact Book!"}).encode("utf-8"))
        elif self.path == "/contacts":
            contacts = self.contact_book.load_contacts()
            self._set_headers()
            self.wfile.write(json.dumps({"message": "Contact list retrieved!", "data": contacts}).encode("utf-8"))
        else:
            self.send_error(404, "Resource not found")

    def do_POST(self):
        if self.path == "/contacts":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data)
                result = self.contact_book.create_contact(data["name"], data["phone_no"])
                self._set_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except (json.JSONDecodeError, KeyError):
                self.send_error(400, "Invalid JSON or missing fields")

    def do_PUT(self):
        if self.path == "/contacts":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data)
                result = self.contact_book.update_contact(
                    data["id"], data["new_name"], data["new_phone_no"]
                )
                self._set_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except (json.JSONDecodeError, KeyError):
                self.send_error(400, "Invalid JSON or missing fields")

    def do_DELETE(self):
        if self.path == "/contacts":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data)
                result = self.contact_book.delete_contact(data["id"])
                self._set_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except (json.JSONDecodeError, KeyError):
                self.send_error(400, "Invalid JSON or missing fields")

    def do_PATCH(self):
        if self.path == "/contacts":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data)

                # Validate the incoming data
                if "old_name" not in data or "old_phone_no" not in data:
                    self.send_error(400, "Missing required fields: 'old_name' and 'old_phone_no'")
                    return

                # Extract the fields to update
                old_name = data["old_name"]
                old_phone_no = data["old_phone_no"]
                new_name = data.get("new_name")  # Optional field
                new_phone_no = data.get("new_phone_no")  # Optional field

                # Perform the partial update
                result = self.contact_book.partial_update_contact(old_name, old_phone_no, new_name, new_phone_no)
                self._set_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except (json.JSONDecodeError, KeyError):
                self.send_error(400, "Invalid JSON or missing required fields")


import pymysql

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
        except pymysql.MySQLError:
            return {"error":"Error connecting to MySQL"}

    def create_contact(self, name, phone_no):
        if self.db_connection:
            try:
                cursor = self.db_connection.cursor()
                sql_statement = "INSERT INTO contacts (name, phone_no) VALUES (%s, %s)"
                cursor.execute(sql_statement, (name, phone_no))
                self.db_connection.commit()
                return {"message": "Contact added successfully!"}
            except pymysql.MySQLError:
                return {"error": "Error adding contact."}

    def load_contacts(self):
        if self.db_connection:
            try:
                cursor = self.db_connection.cursor()
                cursor.execute("SELECT id, name, phone_no FROM contacts")
                contacts = cursor.fetchall()
                return [{"id": contact[0], "name": contact[1], "phone_no": contact[2]} for contact in contacts]
            except pymysql.MySQLError:
                return {"error": "Error loading contacts."}
        return []

    def update_contact(self, contact_id, new_name, new_phone_no):
        if self.db_connection:
            try:
                cursor = self.db_connection.cursor()
                sql_statement = """
                    UPDATE contacts SET name = %s, phone_no = %s 
                    WHERE id = %s
                """
                cursor.execute(sql_statement, (new_name, new_phone_no, contact_id))
                self.db_connection.commit()
                if cursor.rowcount > 0:
                    return {"message": "Contact updated successfully!"}
                else:
                    return {"message": "No matching contact found to update."}
            except pymysql.MySQLError:
                return {"error": "Error updating contact."}

    def delete_contact(self, contact_id):
        if self.db_connection:
            try:
                cursor = self.db_connection.cursor()
                sql_statement = "DELETE FROM contacts WHERE id = %s"
                cursor.execute(sql_statement, (contact_id,))
                self.db_connection.commit()
                if cursor.rowcount > 0:
                    return {"message": "Contact deleted successfully!"}
                else:
                    return {"message": "No matching contact found to delete."}
            except pymysql.MySQLError:
                return {"error": "Error deleting contact."}
            
    def partial_update_contact(self, old_name, old_phone_no, new_name=None, new_phone_no=None):
        if self.db_connection:
            try:
                cursor = self.db_connection.cursor()
                # Build the SQL dynamically based on provided fields
                updates = []
                values = []

                if new_name:
                    updates.append("name = %s")
                    values.append(new_name)
                if new_phone_no:
                    updates.append("phone_no = %s")
                    values.append(new_phone_no)

                if not updates:
                    return {"message": "No fields provided to update."}

                # Add WHERE clause values
                values.extend([old_name, old_phone_no])

                sql_statement = f"""
                    UPDATE contacts 
                    SET {', '.join(updates)} 
                    WHERE name = %s AND phone_no = %s
                """
                cursor.execute(sql_statement, tuple(values))
                self.db_connection.commit()

                if cursor.rowcount > 0:
                    return {"message": "Contact updated successfully!"}
                else:
                    return {"message": "No matching contact found to update."}
            except pymysql.MySQLError as e:
                return {"error": str(e)}
