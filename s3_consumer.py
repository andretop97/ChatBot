import io
import os
import json
import boto3
import PyPDF2
import pandas as pd
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

def listar_pdfs_recursivamente(bucket, prefix=''):
    s3 = boto3.client('s3')
    """Lista todos os arquivos PDF em um bucket (e subpastas) recursivamente."""
    try:
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        pdfs = []
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.lower().endswith('.pdf'):  # Verifica a extensão (case-insensitive)
                        pdfs.append(key)
        return pdfs

    except Exception as e:
        print(f"Erro ao listar PDFs: {e}")
        return []

def get_bucket_object(Bucket, Key):
    s3 = boto3.client('s3')
    try:
        response = s3.get_object(Bucket=Bucket, Key=Key)
        return response['Body'].read()

    except Exception as e:
        print(f"Erro ao obter registro de certificado: {e}")
        return None
    
def extract_text_from_pdf(pdf_data):
    try:
        pdf = PyPDF2.PdfReader(io.BytesIO(pdf_data))
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
        return text

    except Exception as e:
        print(f"Erro ao extrair texto do PDF: {e}")
        return None
    
def get_prompt_template():
    prompt_template = """ 
Você é um assistente de IA especialista em extração de informações de textos de certificados de cursos. Extraia as seguintes informações:
    - Nome do curso
    - Instituição de ensino
    - Área do curso
    - Nível
    - Data de finalização
    - Palavras-chave relacionadas ao curso
Texto do certificado:

{text}

Forneça a saída em formato JSON:
   "Certificado": "Nome do curso",
   "Instituicao": "Instituição de ensino",
   "Area": "Área do curso",
   "Nivel": "Nivel",
   "Data_fim: "Data de finalização no formato dd/mm/aaaa",
   "KeyWords: ["Palavras-chave1", "Palavras-chave2"]
"""
    return PromptTemplate(input_variables=["text"], template=prompt_template)

def get_informacoes_do_certificado(chain , Bucket, pdfKey):
    pdf_data = get_bucket_object(Bucket, pdfKey)
    pdf_text = extract_text_from_pdf(pdf_data)
    try:
        data = chain.invoke({"text": pdf_text})
        return JsonOutputParser().parse(data.content)
    except Exception as e:
        print(f"Erro ao extrair informações do certificado: {e}")
        return None


def atualizar_df(csv_data, Bucket, pdfs, chain):
    df = pd.read_csv(io.StringIO(csv_data))
    for pdf in pdfs:
        if pdf not in df['URI'].values:
            response = get_informacoes_do_certificado(chain, Bucket, pdf)
            if response:
                response['URI'] = pdf
                new_df = pd.DataFrame([response])
                df = pd.concat([df, new_df], ignore_index=True)
                print(df)
    return df

def salvar_registro_atualizado(df, Bucket, registroKey):
    csv_data = df.to_csv(index=False, encoding='utf-8')
    s3 = boto3.client('s3')
    try:
        s3.put_object(Bucket=Bucket, Key=registroKey, Body=csv_data)
        print(f"Registro atualizado com sucesso.")
    except Exception as e:
        print(f"Erro ao salvar registro atualizado: {e}")
    
def atualizar_registro_de_certificados(Bucket, Prefix, registroKey, chain):
    pdfs = listar_pdfs_recursivamente(Bucket, Prefix)
    print(f"Encontrados {len(pdfs)} PDFs.")
    csv_data = get_bucket_object(Bucket, registroKey)
    print(csv_data)
    if csv_data:
        csv_data = csv_data.decode('utf-8')
        r_atualizado = atualizar_df(csv_data, Bucket, pdfs, chain)
        salvar_registro_atualizado(r_atualizado, Bucket, registroKey)
    # atualizar_registro(csv_data, Bucket, pdfs, chain)

# def criar_csv():
#     colunas = ['Certificado', 'Instituicao', 'Area', 'Nivel', 'Data_fim', 'URI', 'KeyWords']
#     df = pd.DataFrame(columns=colunas)
#     return df.to_csv("certifications_file.csv", index=False, encoding='utf-8')

if __name__ == "__main__":
    load_dotenv()    
    Bucket=os.getenv('BUCKET')
    Prefix=os.getenv('PREFIX')
    uri=os.getenv('CERTIFICATION_FILE')


    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.5,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )

    prompt = get_prompt_template()
    chain = prompt | llm
    atualizar_registro_de_certificados(Bucket, Prefix, uri, chain)