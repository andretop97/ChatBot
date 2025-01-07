import os
import boto3
import pandas as pd

from io import BytesIO
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langchain_core.prompts.prompt import PromptTemplate

class Document(BaseModel):
    keyWords: list[str] = Field(description="keywords", title="KeyWords")


def get_bucket_object(Bucket, Key):
    s3 = boto3.client('s3')
    try:
        response = s3.get_object(Bucket=Bucket, Key=Key)
        return response['Body']

    except Exception as e:
        print(f"Erro ao obter registro de carreira: {e}")
        return None  
     
def load_data_from_s3(bucket: str, key:str)-> pd.DataFrame:
    
    try:
        data = get_bucket_object(bucket, key)
        df = pd.read_csv(data, sep=',', encoding='utf-8')
        df['KeyWords'] = df['KeyWords'].astype(str)
        return df
    except Exception as e:
        print(f"Erro ao carregar os dados do S3: {e}")
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

def get_structured_llm():
    modelName = "llama3.2"

    llm = ChatOllama(
        model=modelName,
        temperature=0.6,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    structured_llm = llm.with_structured_output(Document)
    return structured_llm
    

def extract_key_words_from_career_data(df: pd.DataFrame) -> pd.DataFrame:
    structured_llm = get_structured_llm()
    prompt = get_prompt_template()
    chain = prompt | structured_llm

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

    return df

def upload_file_to_s3(bucket: str, key: str, df: pd.DataFrame) -> None:
    s3 = boto3.client('s3')
    try:
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        response = s3.upload_fileobj(csv_buffer, bucket, key)
        return response

    except Exception as e:
        print(f"Erro ao fazer upload do arquivo para o S3: {e}")
        return None
    
if __name__ == "__main__":
    load_dotenv()    
    BUCKET = os.getenv('BUCKET')
    CSV_PATH= os.getenv('CSV_KEY')
    file_name = 'Carreira.csv'

    key = f'{CSV_PATH}/{file_name}'
    print(key)

    df = load_data_from_s3(BUCKET, key)
    df = extract_key_words_from_career_data(df)
    upload_file_to_s3(BUCKET, key, df)