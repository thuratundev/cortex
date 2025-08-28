import psycopg2
from psycopg2.extras import execute_values
from app_config import DB_PARAMS

class db_agent:
    def __init__(self):
        self.connection = psycopg2.connect(**DB_PARAMS)
        

    def insert_document(self, documents: list[dict]):
        try:
            self.cursor = self.connection.cursor()
            query = "INSERT INTO reportvector (id, reportname, description, keywords, samplequeries, modelname, verno, embeddedvector) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
            
            for document in documents:
                values = (
                    document.get("id"),
                    document.get("reportname"),
                    document.get("description"),
                    document.get("keywords"),
                    document.get("samplequeries"),
                    document.get("modelname"),
                    document.get("verno"),
                    document.get("embeddedvector")
                )
                self.cursor.execute(query, values)
            
            self.connection.commit()
        except Exception as ex:
            print(f"Error inserting document: {ex}")
            self.connection.rollback()
        finally:
            self.cursor.close()

    def get_similar_documents(self, embedded_user_query: str) -> list[dict]:
        try:
            self.cursor = self.connection.cursor()
            query = "SELECT reportname,description,1 - (embeddedvector <=> %s::vector) AS similarity FROM reportvector ORDER BY similarity DESC LIMIT 5"
            self.cursor.execute(query, (embedded_user_query,))
            results = self.cursor.fetchall()
            # Convert tuple results to dictionaries with named keys
            documents = []
            for row in results:
                documents.append({
                    'reportname': row[0],
                    'description': row[1],
                    'similarity': row[2]
                })
            return documents
        except Exception as ex:
            print(f"Error retrieving similar documents: {ex}")
            return []
        finally:
            self.cursor.close()

    def close(self):
        self.connection.close()