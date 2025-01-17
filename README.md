# Amazon Return Policy ChatBot
This project provides two implementations of a chatbot use ChatOpenAI and LangChain that can help users with questions about Amazon's return policy. One implementation is a Telegram bot, and the other is a web-based interface using Streamlit.

## Features
 - Answers questions related to Amazon's return policy.
 - Redirects non-return policy queries to Amazon's customer support.
 - Provides initial greeting to users.


## Technologies Used
* **Python**: The core programming language used for development.
* **LangChain**: A library to build and operate with large language models (LLMs).
* **OpenAI** **API**: Used to integrate GPT-3.5 for generating responses.
* **Streamlit**: A framework for building web applications.
* **Python Telegram Bot**: A library to interact with the Telegram Bot API.
* **FAISS**: A library for efficient similarity search and clustering of dense vectors.
* **BeautifulSoup**: A library for web scraping purposes (through **WebBaseLoader in LangChain**).

### **_Two implementations:_** 
Telegram bot and Streamlit app.


### Setup and Installation
 - Prerequisites
 - Python 3.8+
 - pip (Python package installer)
 - Telegram bot token (obtain from BotFather)
 - OpenAI API key (obtain from OpenAI)
 - 
### **Clone the Repository**

```
git clone https://github.com/Yevheniia-Ilchenko/ChatBotAmazon.git

```
### Create a .env File
Create a .env file in the root directory of the project and add the following content:
```
OPENAI_API_KEY=your-openai-api-key
USER_AGENT=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36
TELEGRAM_TOKEN=your-telegram-bot-token

```
### Install Dependencies

```
pip install -r requirements.txt
```
### Running the Telegram Bot
To run the Telegram bot:

Ensure your .env file is correctly set up with your Telegram bot token and OpenAI API key.

Run the telegram_bot.py script:
```
python telegram_bot.py

```

### Running the Streamlit App
To run the Streamlit app:

Ensure your .env file is correctly set up with your OpenAI API key and user agent.

Run the streamlit_bot.py script:

```
streamlit run streamlit_bot.py

```
## Usage
### Telegram Bot
- Start the bot by running the telegram_bot.py script.
 - Open Telegram and search for your bot.
 - Start a conversation with the bot using the /start command.
 - Ask questions related to Amazon's return policy.

![home page](static/img/Screenshot_telegram.jpg)

### Streamlit App
 - Start the app by running the streamlit_bot.py script.
 - Open a web browser and go to the provided localhost URL.
 - Interact with the chatbot by typing your questions in the input field and clicking "Send".


![home page](static/img/Screenshot_streamlit.jpg)