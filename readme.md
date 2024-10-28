**Chat with Amara, your personal chatbot dedicated for sentiment detection**

There are two chatbots. One integrated with llama 2 7-b and the other with Qwen v2 1.5B.
There is also a python file which contains the API of the Qwen Model.

The Functionalities are mentioned below.

* User authentication:

  * Login and registration system
  * Session management
  * Password hashing for security
* Chats are now associated with specific users
* Chat history is filtered by user_id
* Added context from recent conversations
* The chats are stored in the sqlite db along with the sentiment score and they are retrieved each time the user logs in. So the chats are more personalized.

If you have Cuda installed, then comment the line  device='cpu'.
