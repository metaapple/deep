import requests
import base64


def encode_image(image_path):
    """이미지 파일을 Base64 문자열로 변환합니다."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def test_llama_vision(image_path, prompt="이 이미지에 무엇이 보이나요?"):
    url = "http://localhost:11434/api/generate"

    # 이미지를 base64로 변환
    base64_image = encode_image(image_path)

    payload = {
        "model": "llama3.2-vision",
        # "model" : "moondream", # --> 더 경량
        "prompt": prompt,
        "stream": False,
        "images": [base64_image]  # 이미지 데이터를 리스트 형태로 전달
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        print(f"🤖 모델 답변:\n{result.get('response')}")
    except Exception as e:
        print(f"❌ 에러 발생: {e}")


if __name__ == "__main__":
    # 프로젝트 폴더에 있는 이미지 파일명을 넣으세요.
    test_llama_vision("test_image.jpg")