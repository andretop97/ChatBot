import os
import boto3
import pandas as pd
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from dotenv import load_dotenv
from pydantic import BaseModel, Field

class Document(BaseModel):
    keyWords: list[str] = Field(description="keywords", title="KeyWords")


def get_bucket_object(Bucket, Key):
    s3 = boto3.client('s3')
    try:
        response = s3.get_object(Bucket=Bucket, Key=Key)
        return response['Body'].read()

    except Exception as e:
        print(f"Erro ao obter registro de certificado: {e}")
        return None   
    
def get_prompt_template():
    prompt_template = """ 
Você é um assistente de IA especialista em extração de palavras chave referentes a capaciades profissionais.
Extraia as palavras chave do texto fornecido abaixo:

Cargo: {cargo}
Empresa: {empresa}
Modalidade: {modalidade}
Atribuições: {atribuicoes}
Projetos: {projetos}

A saida deve ser convertida em json.
"""
    return PromptTemplate(input_variables=["cargo","empresa","modalidade","atribuicoes","projetos"], template=prompt_template)

if __name__ == "__main__":
    load_dotenv()    
    Bucket=os.getenv('BUCKET')
    Prefix=os.getenv('PREFIX')
    uri=os.getenv('CERTIFICATION_FILE')
    modelName = "llama3.2"


    llm = ChatOllama(
        model=modelName,
        temperature=0.6,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    structured_llm = llm.with_structured_output(Document)

    prompt = get_prompt_template()
    chain = prompt | structured_llm

    df = pd.read_csv('Carreira.csv', sep=',', encoding='utf-8')
    df['KeyWords'] = df['KeyWords'].astype(str)

    for index, row in df.iterrows():
        dict_doc = {}
        dict_doc['cargo'] = row['Cargo']
        dict_doc['empresa'] = row['Empresa']
        dict_doc['modalidade'] = row['Modelidade']
        dict_doc['atribuicoes'] = row['Atribuições']
        dict_doc['projetos'] = row['Projetos']
        response = chain.invoke(dict_doc)

        if response:
            string_lista = ", ".join(str(item) for item in response.keyWords)
            string_final = f"[{string_lista}]"
            df.loc[index, 'KeyWords'] = string_final
            
    df.to_csv('Carreira_new.csv', sep=',', encoding='utf-8', index=False)