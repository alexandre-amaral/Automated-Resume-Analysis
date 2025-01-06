__author__ = 'Alexandre Amaral'

# Importing necessary libraries
import aiohttp
import asyncio
from pymongo import MongoClient

# Connecting to local MongoDB
client = MongoClient('mongodb://localhost:27017')
db = client['analise-automatizada-curriculos']
collection = db['cvs']

# TODO:
'''
Connection to MongoDB Atlas in the cloud

server_client = MongoClient('mongodb+srv://alexandre:<password>@cv-analyzer-database.fbxfj.mongodb.net/')
server_db = server_client['analise-automatizada-curriculos']
server_collection = server_db['curriculos']
'''

# DOCS:
'''
Function to perform asynchronous API requests
'''
async def fetch(session, url):
    # API request
    async with session.get(url) as response:
        # Convert response to JSON
        data = await response.json()
        # Check response status
        if response.status == 200:
            # Insert document into MongoDB
            collection.insert_one(data)
            print('Document inserted successfully')
        else:
            print('API request failed')

# DOCS:
'''
Function to perform asynchronous API requests
'''
async def request(i, f):
    
    async with aiohttp.ClientSession() as session:

        # Base URL of the API
        url_base = 'https://api.candidato.bne.com.br/api/v1/Curriculum/MinData/'

        # List of tasks
        tasks = []

        '''
        Loop to create request tasks based on the range
        determined by the start and end values passed as parameters
        '''
        for num_pag in range(i, f):
            url = url_base + str(num_pag)
            tasks.append(asyncio.create_task(fetch(session, url)))
        await asyncio.gather(*tasks)

# TODO:
'''
Function to update MongoDB Atlas in the cloud with local documents
def update():
    documents = collection.find()
    documents_json = []

    for document in documents:
        document.pop('_id', None)
        document_json = json.loads(json_util.dumps(document))
        documents_json.append(document_json)

    server_collection.insert_many(documents_json)
    print('Documents inserted in Atlas')
'''

# DOCS:
'''
Main function to execute the program with requests
in intervals of 10,000 to avoid API timeout errors
'''
async def main():
    for i in range(0, 1000):
        start_value = 10000 * (i-1)
        end_value = 10000 * i
        await request(start_value, end_value)

# DOCS:
'''
Function to execute the main program asynchronously with asyncio
'''
async def run():
    try:
        await main()

        # TODO:
        '''
        Function to update MongoDB Atlas in the cloud
        with local documents
        update()
        '''

    except Exception as e:
        print(e)

# DOCS:
'''
Execute the main program asynchronously with asyncio
'''
asyncio.run(run())
