# HTML파일을 템플릿으로 사용하기 위한 모듈 임포트
from flask import Flask, render_template, request
# 화면이동, 세션처리 등을 위한 모듈 임포트
from flask import redirect, session, url_for
# 문자열 깨짐 방지를 위한 인코딩 처리를 위한 모듈 임포트
from markupsafe import escape

# 플라스크 앱 초기화
app = Flask(__name__)


# 앱을 최초로 실행했을때의 화면. 주로 index화면이라고 한다.
@app.route('/')
def root():
    return 'Hello Flask Apps'

@app.route('/visual')
def show_visual():
    return render_template('visual.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# Page not found 에러 발생시 핸들링
@app.errorhandler(404)
def page_not_found(error):
    print("오류 로그:", error)  # 서버콘솔에 출력
    return render_template('404.html')
    return "페이지가 없습니다. URL를 확인 하세요", 404


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
