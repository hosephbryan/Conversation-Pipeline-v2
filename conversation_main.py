import asyncio


from DetectEmotions import listen_for_commands
from rag_utils import prepare_and_split_docs, ingest_into_vectordb


async def main():
    file_directory="Conversation-history"
    embedding_model='sentence-transformers/all-MiniLM-L6-v2'
    #embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    past_chat_history = prepare_and_split_docs(file_directory)
    vectorstore = ingest_into_vectordb(past_chat_history, embeddings)
    retriever = vectorstore.as_retriever()
    model = 'yi:6b-chat-q4_K_M'

    chat_chain = await initialize_conversation_chain(model, retriever)

    while True:
        user_input = None
        stt_input = None   

        while user_input is None:                                                                                                             
            user_input, stt_input = await listen_for_commands()

        if stt_input.lower() == "exit" or stt_input.lower() == "goodbye":
            print("AI Therapist: Take care! Feel free to reach out anytime.")
            break
        
        response_chain =  chat_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "001-1"}}
        )
        
        ai_response = response_chain["answer"]
        print(f"AI Therapist: {ai_response}")


if __name__ == "__main__":
    asyncio.run(main())                                                                   