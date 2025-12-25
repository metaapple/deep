import requests
import base64


def encode_image(image_path):
    """이미지 파일을 Base64 문자열로 변환합니다."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def extract_text_from_image(image_path,
                            prompt="이 이미지에 있는 모든 텍스트를 정확하게 추출해 주세요. 글자가 왜곡되거나 흐릿하더라도 최대한 원문에 가깝게 작성해 주세요."):
    """
    Ollama의 Vision 모델을 사용해 이미지에서 텍스트를 OCR처럼 추출합니다.
    """
    url = "http://localhost:11434/api/generate"

    # 이미지를 base64로 인코딩
    base64_image = encode_image(image_path)

    payload = {
        "model": "moondream",  # 또는 "llava", "llama3.2-vision" 등 사용 중인 모델
        "prompt": prompt,
        "stream": False,
        "images": [base64_image]
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()

        extracted_text = result.get('response', '').strip()

        # 추출된 텍스트를 화면에 출력
        print("=" * 50)
        print("OCR로 추출된 텍스트:")
        print("=" * 50)
        print(extracted_text)
        print("=" * 50)

        return extracted_text

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP 에러 발생: {http_err}")
        print(f"응답 내용: {response.text}")
    except Exception as e:
        print(f"에러 발생: {e}")


if __name__ == "__main__":
    # 이미지 파일 경로를 자신의 환경에 맞게 수정하세요
    image_file = "./img.png"

    extract_text_from_image(image_file)