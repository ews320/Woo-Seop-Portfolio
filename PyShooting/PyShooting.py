import pygame
import sys
import random
from tkinter import *
from time import sleep
import time

padWidth = 700       # 게임화면의 가로크기
padHeight = 400      # 게임화면의 세로크기
trashImage = ['c:/PyShooting/trash/trash01.png', 'c:/PyShooting/trash/trash02.png', 'c:/PyShooting/trash/trash03.png', 'c:/PyShooting/trash/trash04.png', 'c:/PyShooting/trash/trash05.png', \
             'c:/PyShooting/trash/trash06.png', 'c:/PyShooting/trash/trash07.png', 'c:/PyShooting/trash/trash08.png', 'c:/PyShooting/trash/trash09.png', 'c:/PyShooting/trash/trash10.png', \
             'c:/PyShooting/trash/trash11.png', 'c:/PyShooting/trash/trash12.png', 'c:/PyShooting/trash/trash13.png', 'c:/PyShooting/trash/trash14.png', 'c:/PyShooting/trash/trash15.png', \
             'c:/PyShooting/trash/trash16.png', 'c:/PyShooting/trash/trash17.png', 'c:/PyShooting/trash/trash18.png']
fishImage = ['c:/PyShooting/fish/fish01.png', 'c:/PyShooting/fish/fish02.png', 'c:/PyShooting/fish/fish03.png', 'c:/PyShooting/fish/fish04.png', 'c:/PyShooting/fish/fish05.png', \
             'c:/PyShooting/fish/fish06.png', 'c:/PyShooting/fish/fish07.png', 'c:/PyShooting/fish/fish08.png', 'c:/PyShooting/fish/fish09.png', 'c:/PyShooting/fish/fish10.png', \
             'c:/PyShooting/fish/fish11.png', 'c:/PyShooting/fish/fish12.png', 'c:/PyShooting/fish/fish13.png', 'c:/PyShooting/fish/fish14.png', 'c:/PyShooting/fish/fish15.png', \
             'c:/PyShooting/fish/fish16.png']
fishImage2 = ['c:/PyShooting/fish/fish01.png', 'c:/PyShooting/fish/fish02.png', 'c:/PyShooting/fish/fish03.png', 'c:/PyShooting/fish/fish04.png', 'c:/PyShooting/fish/fish05.png', \
             'c:/PyShooting/fish/fish06.png', 'c:/PyShooting/fish/fish07.png', 'c:/PyShooting/fish/fish08.png', 'c:/PyShooting/fish/fish09.png', 'c:/PyShooting/fish/fish10.png', \
             'c:/PyShooting/fish/fish11.png', 'c:/PyShooting/fish/fish12.png', 'c:/PyShooting/fish/fish13.png', 'c:/PyShooting/fish/fish14.png', 'c:/PyShooting/fish/fish15.png', \
             'c:/PyShooting/fish/fish16.png'] 
sharkImage = ['c:/PyShooting/shark.png']
shark2Image = ['c:/PyShooting/shark2.png']
trashSound = ['c:/PyShooting/Sound/trashSound.mp3']
fishSound = ['c:/PyShooting/Sound/fishSound.mp3']



score = 0
bnetCount = 0

# 종료 함수
def gameClose():
    pygame.quit()
    sys.exit()
    
#esc키 게임 종료여부
def gameCloseChoice():
    window = Tk()
    window.title("game close")
    def gameCloseButton():
        window.destroy()
        gameClose()                           
    
    #종료 여부 윈도우 만들기
    btnClose = Button(window, text = "close",command = gameCloseButton,width=30,height=3)
    btnCtn = Button(window, text = "continue",command = window.destroy,width=30,height=3)
    btnClose.pack()
    btnCtn.pack()
    window.mainloop()
    
white = (255, 255, 255)

titleImg = pygame.image.load("c:/PyShooting/게임이름.png")
normalImg = pygame.image.load("c:/PyShooting/normal.png")
hardImg = pygame.image.load("c:/PyShooting/hard.png")
clicknormalImg = pygame.image.load("c:/PyShooting/normal.png")
clickhardImg = pygame.image.load("c:/PyShooting/hard.png")
restartImg = pygame.image.load("c:/PyShooting/재시작.png")
menuImg = pygame.image.load("c:/PyShooting/메뉴.png")

gameDisplay = pygame.display.set_mode((padWidth, padHeight))


class Button1:
    def __init__(self, img_in, x, y, width, height, img_act, x_act, y_act, action = None):
        mouse = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()
        if x + width > mouse[0] > x and y + height > mouse[1] > y:
            gameDisplay.blit(img_act,(x_act, y_act))
            if click[0] and action != None:
                time.sleep(1)
                action()
        else:
            gameDisplay.blit(img_in,(x,y))

# 총 점 계산
def writeScore(count):
    global gamePad
    font = pygame.font.Font('c:/PyShooting/font.ttf', 20)
    text = font.render('총 점: ' + str(count), True, (255, 255, 255))
    gamePad.blit(text, (10, 0))

# 쓰레기가 통과한 개수
def writePassed(count):
    global gamePad
    font = pygame.font.Font('c:/PyShooting/font.ttf', 20)
    text = font.render('놓친 쓰레기: ' + str(count), True, (255, 0, 0))
    gamePad.blit(text, (360, 0))

def bnetCnt():
    global gamePad, bnetScore, bnetCount
    if bnetScore >= 500:
        bnetCount += 1
        bnetScore -= 500
    font = pygame.font.Font('c:/PyShooting/font.ttf', 20)
    text = font.render('필살기: ' + str(bnetCount), True, (255, 0, 0))
    gamePad.blit(text, (360, 20))

# 게임시작 메시지 출력
def start():
    global gamePad, background
    menu = True
    pygame.mixer.music.stop()     # 배경 음악 정지
        
    while menu:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        drawObject(background, 0, 0)  
        
        titletext = gameDisplay.blit(titleImg, (150,50))
        normalButton = Button1(normalImg,90,260,295,88,clicknormalImg,80,258,runGame)
        hardButton = Button1(hardImg,410,260,199,89,clickhardImg,400,258,runGame2)
        pygame.display.update()
        clock.tick(15)        

# 게임오버 메시지 출력
def gameOver():
    global gamePad, background
    menu = True
    pygame.mixer.music.stop()     # 배경 음악 정지
    gameOverSound.play()          # 게임 오버 사운드 재생
    while menu:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        drawObject(background, 0, 0)  
        font = pygame.font.Font('c:/PyShooting/font.ttf', 80)
        text = font.render('미션 실패!!', True, (255, 0, 0))
        textpos = text.get_rect()
        textpos.center = (padWidth/2, padHeight/4)
        gamePad.blit(text, textpos)
        font = pygame.font.Font('c:/PyShooting/font.ttf', 50)
        text = font.render('총 점: '+ str(score), True, (255, 255, 255))
        text1 = text.get_rect()
        text1.center = (padWidth/2, padHeight/2)
        gamePad.blit(text, text1)        
        restartButton = Button1(restartImg,200,260,91,52,restartImg,190,258,runGame)
        menuButton = Button1(menuImg,410,260,91,52,menuImg,400,258,start)
        pygame.display.update()
        clock.tick(15)

def gameOver2():
    global gamePad, background
    menu = True
    pygame.mixer.music.stop()     # 배경 음악 정지
    gameOverSound.play()          # 게임 오버 사운드 재생
    while menu:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        drawObject(background, 0, 0)  
        font = pygame.font.Font('c:/PyShooting/font.ttf', 80)
        text = font.render('미션 실패!!', True, (255, 0, 0))
        textpos = text.get_rect()
        textpos.center = (padWidth/2, padHeight/4)
        gamePad.blit(text, textpos)
        font = pygame.font.Font('c:/PyShooting/font.ttf', 50)
        text = font.render('총 점: '+ str(score), True, (255, 255, 255))
        text1 = text.get_rect()
        text1.center = (padWidth/2, padHeight/2)
        gamePad.blit(text, text1)        
        restartButton = Button1(restartImg,200,260,91,52,restartImg,190,258,runGame2)
        menuButton = Button1(menuImg,410,260,70,52,menuImg,400,258,start)
        pygame.display.update()
        clock.tick(15)

# 게임 클리어 메시지 출력
def clear():
    global gamePad, background
    menu = True
    pygame.mixer.music.stop()     # 배경 음악 정지
    clearSound.play()             # 게임 오버 사운드 재생
    while menu:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        drawObject(background, 0, 0)  
        font = pygame.font.Font('c:/PyShooting/font.ttf', 80)
        text = font.render('클리어!!', True, (0, 0, 255))
        textpos = text.get_rect()
        textpos.center = (padWidth/2, padHeight/4)
        gamePad.blit(text, textpos)
        font = pygame.font.Font('c:/PyShooting/font.ttf', 50)
        text = font.render('총 점: '+ str(score), True, (255, 255, 255))
        text1 = text.get_rect()
        text1.center = (padWidth/2, padHeight/2)
        gamePad.blit(text, text1)     
        restartButton = Button1(restartImg,200,260,91,52,restartImg,190,258,runGame)
        menuButton = Button1(menuImg,410,260,70,52,menuImg,400,258,start)
        pygame.display.update()
        clock.tick(15)

def clear2():
    global gamePad, background
    menu = True
    pygame.mixer.music.stop()     # 배경 음악 정지
    clearSound.play()             # 게임 오버 사운드 재생
    while menu:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        drawObject(background, 0, 0)  
        font = pygame.font.Font('c:/PyShooting/font.ttf', 80)
        text = font.render('클리어!!', True, (0, 0, 255))
        textpos = text.get_rect()
        textpos.center = (padWidth/2, padHeight/4)
        gamePad.blit(text, textpos)
        font = pygame.font.Font('c:/PyShooting/font.ttf', 50)
        text = font.render('총 점: '+ str(score), True, (255, 255, 255))
        text1 = text.get_rect()
        text1.center = (padWidth/2, padHeight/2)
        gamePad.blit(text, text1)     
        restartButton = Button1(restartImg,200,260,91,52,restartImg,190,258,runGame2)
        menuButton = Button1(menuImg,410,260,70,52,menuImg,400,258,start)
        pygame.display.update()
        clock.tick(15)
        
    
# 게임에 등장하는 객체를 드로잉
def drawObject(obj, x, y):
    global gamePad
    gamePad.blit(obj, (x, y))

def drawObject2(obj, x, y):
    global gamePad
    gamePad.blit(obj, (x, y))

def initGame():
    global gamePad, clock, fighter, missile, missileSound, gameOverSound, background, clearSound, trashMissSound, plus, minus, bignet
    pygame.init() 
    gamePad = pygame.display.set_mode((padWidth, padHeight))
    pygame.display.set_caption('해양환경보호 Shooting game')                 # 게임 이름
    background = pygame.image.load('c:/PyShooting/background.png')           # 배경 그림
    fighter = pygame.image.load('c:/PyShooting/diver.png')                   # 사람 그림
    missile = pygame.image.load('c:/PyShooting/net.png')                     # 그물 그림
    bignet = pygame.image.load('c:/PyShooting/bignet.png')                   # 필살기 그림
    plus = pygame.image.load('c:/PyShooting/+.png')                          # +100 그림
    minus = pygame.image.load('c:/PyShooting/-.png')                         # -100 그림
    pygame.mixer.music.load('c:/PyShooting/Sound/music.mp3')                 # 배경 음악 
    pygame.mixer.music.play(-1)                                              # 배경 음악 재생
    missileSound = pygame.mixer.Sound('c:/PyShooting/Sound/net.mp3')         # 그물 사운드
    gameOverSound = pygame.mixer.Sound('c:/PyShooting/Sound/gameover.wav')   # 게임 오버 사운드
    clearSound = pygame.mixer.Sound('c:/PyShooting/Sound/clearSound.mp3')    # 클리어 음악
    trashMissSound = pygame.mixer.Sound('c:/PyShooting/Sound/trashMiss.mp3') # 쓰레기 놓침 사운드
    clock = pygame.time.Clock()


def runGame():
    global gamepad, clock, fighter, missile, missileSound, background, score, clearSound, trashMissSound, plus, minus, bnetCount, bignet, bnetScore, bnetCount
    pygame.mixer.music.play(-1)   # 배경 음악 재생
    score = 0
    bnetCount = 1
    bnetScore = 0

    # 사람 크기
    fighterSize = fighter.get_rect().size
    fighterWidth = fighterSize[0]
    fighterHeight = fighterSize[1]

    # 사람 초기 위치 (x,y)
    x = padWidth * 0.05
    y = padHeight * 0.8
    fighterX = 0
    fighterY = 0

    # 무기 좌표 리스트
    missileXY = []

    # 무기 좌표 리스트
    bignetXY = []

    # 쓰레기 랜덤 생성
    trash = pygame.image.load(random.choice(trashImage))
    trash = pygame.transform.rotate(trash,random.randrange(0,180))
    trashSize = trash.get_rect().size
    trashWidth = trashSize[1]
    trashHeight = trashSize[0]
    destroySound = pygame.mixer.Sound(random.choice(trashSound))

    # 쓰레기 초기 위치 설정
    trashX = padWidth
    trashY = random.randrange(0, padHeight - trashHeight)
    trashSpeed = 2

    # 물고기 랜덤 생성
    fish = pygame.image.load(random.choice(fishImage))
    fishSize = fish.get_rect().size
    fishWidth = fishSize[1]
    fishHeight = fishSize[0]
    CatchSound = pygame.mixer.Sound(random.choice(fishSound))

    fish2 = pygame.image.load(random.choice(fishImage2))  
    fish2Size = fish2.get_rect().size
    fish2Width = fish2Size[1]
    fish2Height = fish2Size[0]
    CatchSound = pygame.mixer.Sound(random.choice(fishSound))

    # 물고기 초기 위치 설정
    fishX = padWidth
    fishY = random.randrange(0, padHeight - fishHeight)
    fishSpeed = 3

    fish2X = padWidth 
    fish2Y = random.randrange(0, padHeight - fish2Height)
    fish2Speed = 2.5

    # 상어 랜덤 생성
    shark = pygame.image.load(random.choice(sharkImage))
    sharkSize = shark.get_rect().size
    sharkWidth = sharkSize[1]
    sharkHeight = sharkSize[0]
    destroySound = pygame.mixer.Sound(random.choice(trashSound))

    # 상어 초기 위치 설정
    sharkX = padWidth
    sharkY = random.randrange(0, padHeight - sharkHeight)
    sharkSpeed = 4

    # 사람 그물에 쓰레기가 맞았을 경우 True
    isShot = False
    shotCount = 0
    trashPassed = 0

    # 사람 그물에 물고기가  맞았을 경우 True
    Shot = False
    fishCount = 0

    Shot2 = False 
    fish2Count = 0

    onGame = False
    while not onGame:
        for event in pygame.event.get():
            if event.type in [pygame.QUIT]:        # 게임 프로그램 종료
                gameClose()

            if event.type in [pygame.KEYDOWN]:
                if event.key == pygame.K_LEFT:     # 사람 왼쪽으로 이동
                    fighterX -= 5

                elif event.key == pygame.K_RIGHT:  # 사람 오른쪽으로 이동
                    fighterX += 5

                elif event.key == pygame.K_UP:     # 사람 위쪽으로 이동
                    fighterY -= 5

                elif event.key == pygame.K_DOWN:   # 사람 아래로 이동
                    fighterY += 5

                elif event.key == pygame.K_SPACE:  # 그물 발사
                    missileSound.play()            # 그물 사운드 재생
                    missileX = x + fighterWidth
                    missileY = y + fighterHeight/5
                    missileXY.append([missileX, missileY])

                elif event.key == pygame.K_ESCAPE: # esc 키로 게임 종료
                    gameCloseChoice()

                elif event.key == pygame.K_LCTRL:  #필살기 발사
                    if bnetCount > 0:
                        bnetCount -= 1
                        missileSound.play()            # 그물 사운드 재생
                        bignetX = x + fighterWidth
                        bignetY = y + fighterHeight-80
                        bignetXY.append([bignetX, bignetY])
                    else:
                        bnetCount = 0

            if event.type in [pygame.KEYUP]:
                if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                    fighterX = 0
                elif event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                    fighterY = 0

        drawObject(background, 0, 0)               # 배경 화면 그리기

        # 사람 위치 재조정
        x += fighterX
        if x < 0:
            x = 0
        elif x > padWidth - fighterWidth:
            x = padWidth - fighterWidth

        y += fighterY
        if y < 0:
            y = 0
        elif y > padHeight - fighterHeight:
            y = padHeight - fighterHeight

        # 사람이 상어와 충돌했는지 체크
        if x + fighterWidth >= sharkX and x <= sharkX + sharkWidth :
            if(y + fighterHeight >= sharkY * 1.2 and y <= sharkY + sharkHeight * 0.6):
                gameOver()

        drawObject(fighter, x, y)                # 사람을 게임 화면의 (x,y) 좌표에 그림

        # 그물 발사 화면에 그리기
        if len(missileXY) != 0:
            for i, bxy in enumerate(missileXY):  # 그물 요소에 대해 반복함
                bxy[0] += 10                     # 그물이 y좌표 -10 (위로 이동)
                missileXY[i][0] = bxy[0]

                # 그물이 상어를 맞추었을 경우 그물 제거
                if bxy[0] >= sharkX and bxy[0] <= sharkX + 10:
                    if bxy[1] >= sharkY - 50 and bxy[1] <= sharkY + 50:
                        missileXY.remove(bxy)
                        break

                # 그물이 쓰레기을 맞추었을 경우
                if bxy[0] > trashX and bxy[0] < trashX + trashWidth:
                    if bxy[1] >= trashY - 50 and bxy[1] <= trashY + trashHeight:
                        missileXY.remove(bxy)
                        isShot = True
                        shotCount += 1
                        score += 100
                        bnetScore += 100
                        break

                # 그물이 물고기를 맞추었을 경우
                if bxy[0] > fishX and bxy[0] < fishX + fishWidth:
                    if bxy[1] >= fishY - 40 and bxy[1] <= fishY + fishHeight:
                        missileXY.remove(bxy)
                        Shot = True
                        fishCount += 1
                        score -= 100
                        bnetScore -= 100
                        if score < 0:
                            score = 0
                            bnetScore = 0
                            break
                        if bnetScore < 0:
                            bnetScore = 0
                            break

                if bxy[0] > fish2X and bxy[0] < fish2X + fish2Width:
                    if bxy[1] >= fish2Y - 40 and bxy[1] <= fish2Y + fish2Height:
                        missileXY.remove(bxy)
                        Shot2 = True
                        fish2Count += 1
                        score -= 100
                        bnetScore -= 100
                        if score < 0:
                            score = 0
                            break
                        if bnetScore < 0:
                            bnetScore = 0
                            break


                if bxy[0] >= (padWidth - 25):     # 그물이 화면 밖으로 벗어나면
                    try:
                        missileXY.remove(bxy)     # 그물 제거
                    except:
                        pass

        if len(missileXY) != 0:
            for bx, by in missileXY:
                drawObject(missile, bx, by)


        # 필살기 그물 발사 화면에 그리기
        if len(bignetXY) != 0:
            for i, bxy in enumerate(bignetXY):  # 그물 요소에 대해 반복함
                bxy[0] += 1                     # 그물이 y좌표 -10 (위로 이동)
                bignetXY[i][0] = bxy[0]

                # 그물이 쓰레기을 맞추었을 경우
                if bxy[0] +80 > trashX and bxy[0] < trashX + trashWidth:
                    if bxy[1] + 100 >= trashY - 100 and bxy[1] <= trashY + trashHeight:
                        isShot = True
                        shotCount += 1
                        score += 100
                        break

                if bxy[0] >= (padWidth - 25):     # 그물이 화면 밖으로 벗어나면
                    try:
                        bignetXY.remove(bxy)     # 그물 제거
                    except:
                        pass

        if len(bignetXY) != 0:
            for bx, by in bignetXY:
                drawObject(bignet, bx, by)

        if score == 1000:
            clear()

        trashX -= trashSpeed                      # 쓰레기 왼쪽으로 움직임

        # 쓰레기를 놓친 경우
        if trashX <= 0:
            # 새로운 쓰레기 (랜덤)
            trash = pygame.image.load(random.choice(trashImage))
            trash = pygame.transform.rotate(trash,random.randrange(0,180))
            trashSize = trash.get_rect().size
            trashWidth = trashSize[1]
            trashHeight = trashSize[0]
            trashX = padWidth
            trashY = random.randrange(0, padHeight - trashHeight)
            trashPassed += 1
            if trashPassed < 3:         # 쓰레기 2개까지 사운드 추가
                trashMissSound.play()

        if trashPassed == 3:            # 쓰레기 3개 놓치면 게임오버
            gameOver()

        writeScore(score)

        bnetCnt()

        # 놓친 쓰레기 수 표시
        writePassed(trashPassed)

        fishX -= fishSpeed             # 물고기 움직임
        fish2X -= fish2Speed 
        
        # 물고기가 왼쪽으로 이동한 경우
        if fishX <= 0:
            # 새로운 물고기 (랜덤)
            fish = pygame.image.load(random.choice(fishImage))
            fishSize = fish.get_rect().size
            fishWidth = fishSize[1]
            fishHeight = fishSize[0]
            fishX = padWidth
            fishY = random.randrange(0, padHeight - fishHeight)

        if fish2X <= 0: 
            # 새로운 물고기 (랜덤)
            fish2 = pygame.image.load(random.choice(fishImage2))
            fish2Size = fish2.get_rect().size
            fish2Width = fish2Size[1]
            fish2Height = fish2Size[0]
            fish2X = padWidth
            fish2Y = random.randrange(0, padHeight - fish2Height)

        sharkX -= sharkSpeed             # 상어 움직임
        
        # 상어가 왼쪽으로 이동한 경우
        if sharkX <= 0:
            # 새로운 상어 (랜덤)
            shark = pygame.image.load(random.choice(sharkImage))
            sharkSize = shark.get_rect().size
            sharkWidth = sharkSize[1]
            sharkHeight = sharkSize[1]
            sharkX = padWidth
            sharkY = random.randrange(0, padHeight - sharkHeight)

        # 쓰레기를 맞춘 경우
        if isShot:
            # 쓰레기 폭발
            drawObject(plus, trashX, trashY)       # 쓰레기 폭발 그리기
            destroySound.play()                    # 쓰레기 폭발 사운드 재생

            # 새로운 쓰레기 (랜덤)
            trash = pygame.image.load(random.choice(trashImage))
            trash = pygame.transform.rotate(trash,random.randrange(0,180))
            trashSize = trash.get_rect().size
            trashWidth = trashSize[0]
            trashHeight = trashSize[1]
            trashX = padWidth
            trashY = random.randrange(0, padHeight - trashHeight)
            destroySound = pygame.mixer.Sound(random.choice(trashSound))
            isShot = False

            # 쓰레기 맞추면 속도 증가
            trashSpeed += 0.02
            if trashSpeed >= 10:
                trashSpeed = 10

        # 물고기를 맞춘 경우
        if Shot:
            # 물고기 폭발
            drawObject(minus, fishX, fishY)         # 물고기 폭발 그리기
            CatchSound.play()                       # 물고기 폭발 사운드 재생


            # 새로운 물고기 (랜덤)
            fish = pygame.image.load(random.choice(fishImage))
            fishSize = fish.get_rect().size
            fishWidth = fishSize[0]
            fishHeight = fishSize[1]
            fishX = padWidth
            fishY = random.randrange(0, padHeight - fishHeight)
            CatchSound = pygame.mixer.Sound(random.choice(fishSound))
            Shot = False

        if Shot2: 
            # 물고기 폭발
            drawObject2(minus, fish2X, fish2Y)       # 물고기 폭발 그리기
            CatchSound.play()                        # 물고기 폭발 사운드 재생

            fish2 = pygame.image.load(random.choice(fishImage2)) 
            fish2Size = fish2.get_rect().size
            fish2Width = fish2Size[0]
            fish2Height = fish2Size[1]
            fish2X = padWidth
            fish2Y = random.randrange(0, padHeight - fish2Height)
            CatchSound = pygame.mixer.Sound(random.choice(fishSound))
            Shot2 = False

        drawObject(trash, trashX, trashY)      # 쓰레기 그리기

        drawObject(fish, fishX, fishY)         # 물고기 그리기

        drawObject2(fish2, fish2X, fish2Y)

        drawObject(shark, sharkX, sharkY)      # 물고기 그리기
        
        pygame.display.update()                # 게임화면을 다시그림

        clock.tick(60)                         # 게임화면의 초당 프레임수를 60으로 설정

    pygame.quit()         # pygame 종료




def runGame2():
    global gamepad, clock, fighter, missile, missileSound, background, score, clearSound, trashMissSound, plus, minus
    pygame.mixer.music.play(-1)   # 배경 음악 재생        
    score = 0

    # 사람 크기
    fighterSize = fighter.get_rect().size
    fighterWidth = fighterSize[0]
    fighterHeight = fighterSize[1]

    # 사람 초기 위치 (x,y)
    x = padWidth * 0.05
    y = padHeight * 0.8
    fighterX = 0
    fighterY = 0

    # 무기 좌표 리스트
    missileXY = []

    # 무기 좌표 리스트
    bignetXY = []

    # 쓰레기 랜덤 생성
    trash = pygame.image.load(random.choice(trashImage))
    trash = pygame.transform.rotate(trash,random.randrange(0,180))
    trashSize = trash.get_rect().size
    trashWidth = trashSize[1]
    trashHeight = trashSize[0]
    destroySound = pygame.mixer.Sound(random.choice(trashSound))

    # 쓰레기 초기 위치 설정
    trashX = padWidth
    trashY = random.randrange(0, padHeight - trashHeight)
    trashSpeed = 2

    # 물고기 랜덤 생성
    fish = pygame.image.load(random.choice(fishImage))
    fishSize = fish.get_rect().size
    fishWidth = fishSize[1]
    fishHeight = fishSize[0]
    CatchSound = pygame.mixer.Sound(random.choice(fishSound))

    fish2 = pygame.image.load(random.choice(fishImage2))  
    fish2Size = fish2.get_rect().size
    fish2Width = fish2Size[1]
    fish2Height = fish2Size[0]
    CatchSound = pygame.mixer.Sound(random.choice(fishSound))

    # 물고기 초기 위치 설정
    fishX = padWidth
    fishY = random.randrange(0, padHeight - fishHeight)
    fishSpeed = 3

    fish2X = padWidth 
    fish2Y = random.randrange(0, padHeight - fish2Height)
    fish2Speed = 2.5

    # 상어 랜덤 생성
    shark = pygame.image.load(random.choice(sharkImage))
    sharkSize = shark.get_rect().size
    sharkWidth = sharkSize[1]
    sharkHeight = sharkSize[1]
    destroySound = pygame.mixer.Sound(random.choice(trashSound))

    shark2 = pygame.image.load(random.choice(shark2Image))
    shark2Size = shark2.get_rect().size
    shark2Width = shark2Size[1]
    shark2Height = shark2Size[1]
    destroySound = pygame.mixer.Sound(random.choice(trashSound))

    # 상어 초기 위치 설정
    sharkX = padWidth
    sharkY = random.randrange(0, padHeight - sharkHeight)
    sharkSpeed = 4

    shark2X = padWidth
    shark2Y = random.randrange(0, padHeight - shark2Height)
    shark2Speed = 6

    # 사람 그물에 쓰레기가 맞았을 경우 True
    isShot = False
    shotCount = 0
    trashPassed = 0

    # 사람 그물에 물고기가  맞았을 경우 True
    Shot = False
    fishCount = 0

    Shot2 = False 
    fish2Count = 0

    onGame = False
    while not onGame:
        for event in pygame.event.get():
            if event.type in [pygame.QUIT]:        # 게임 프로그램 종료
                gameClose()

            if event.type in [pygame.KEYDOWN]:
                if event.key == pygame.K_LEFT:     # 사람 왼쪽으로 이동
                    fighterX -= 5

                elif event.key == pygame.K_RIGHT:  # 사람 오른쪽으로 이동
                    fighterX += 5

                elif event.key == pygame.K_UP:     # 사람 위쪽으로 이동
                    fighterY -= 5

                elif event.key == pygame.K_DOWN:   # 사람 아래로 이동
                    fighterY += 5

                elif event.key == pygame.K_SPACE:  # 그물 발사
                    missileSound.play()            # 그물 사운드 재생
                    missileX = x + fighterWidth
                    missileY = y + fighterHeight/5
                    missileXY.append([missileX, missileY])

                elif event.key == pygame.K_ESCAPE: # esc 키로 게임 종료
                    gameCloseChoice()


            if event.type in [pygame.KEYUP]:
                if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                    fighterX = 0
                elif event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                    fighterY = 0

        drawObject(background, 0, 0)               # 배경 화면 그리기

        # 사람 위치 재조정
        x += fighterX
        if x < 0:
            x = 0
        elif x > padWidth - fighterWidth:
            x = padWidth - fighterWidth

        y += fighterY
        if y < 0:
            y = 0
        elif y > padHeight - fighterHeight:
            y = padHeight - fighterHeight

        # 사람이 상어와 충돌했는지 체크
        if x + fighterWidth >= sharkX and x <= sharkX + sharkWidth :
            if(y + fighterHeight >= sharkY * 1.2 and y <= sharkY + sharkHeight * 0.6):
                gameOver2()

        if x + fighterWidth >= shark2X and x <= shark2X + shark2Width :
            if(y + fighterHeight >= shark2Y * 1.2 and y <= shark2Y + shark2Height * 0.6):
                gameOver2()

        drawObject(fighter, x, y)                # 사람을 게임 화면의 (x,y) 좌표에 그림

        # 그물 발사 화면에 그리기
        if len(missileXY) != 0:
            for i, bxy in enumerate(missileXY):  # 그물 요소에 대해 반복함
                bxy[0] += 10                     # 그물이 y좌표 -10 (위로 이동)
                missileXY[i][0] = bxy[0]

                # 그물이 상어를 맞추었을 경우 그물 제거
                if bxy[0] >= sharkX and bxy[0] <= sharkX + 10:
                    if bxy[1] >= sharkY - 50 and bxy[1] <= sharkY + 50:
                        missileXY.remove(bxy)
                        break

                if bxy[0] >= shark2X and bxy[0] <= shark2X + 10:
                    if bxy[1] >= shark2Y - 50 and bxy[1] <= shark2Y + 50:
                        missileXY.remove(bxy)
                        break

                # 그물이 쓰레기을 맞추었을 경우
                if bxy[0] > trashX and bxy[0] < trashX + trashWidth:
                    if bxy[1] >= trashY - 50 and bxy[1] <= trashY + trashHeight:
                        missileXY.remove(bxy)
                        isShot = True
                        shotCount += 1
                        score += 100
                        break

                # 그물이 물고기를 맞추었을 경우
                if bxy[0] > fishX and bxy[0] < fishX + fishWidth:
                    if bxy[1] >= fishY - 40 and bxy[1] <= fishY + fishHeight:
                        missileXY.remove(bxy)
                        Shot = True
                        fishCount += 1
                        score -= 100
                        if score < 0:
                            score = 0
                            break

                if bxy[0] > fish2X and bxy[0] < fish2X + fish2Width:
                    if bxy[1] >= fish2Y - 40 and bxy[1] <= fish2Y + fish2Height:
                        missileXY.remove(bxy)
                        Shot2 = True
                        fish2Count += 1
                        score -= 100
                        if score < 0:
                            score = 0
                            break


                if bxy[0] >= (padWidth - 25):     # 그물이 화면 밖으로 벗어나면
                    try:
                        missileXY.remove(bxy)     # 그물 제거
                    except:
                        pass

        if len(missileXY) != 0:
            for bx, by in missileXY:
                drawObject(missile, bx, by)


        if score == 1000:
            clear2()

        trashX -= trashSpeed                      # 쓰레기 왼쪽으로 움직임

        # 쓰레기를 놓친 경우
        if trashX <= 0:
            # 새로운 쓰레기 (랜덤)
            trash = pygame.image.load(random.choice(trashImage))
            trash = pygame.transform.rotate(trash,random.randrange(0,180))
            trashSize = trash.get_rect().size
            trashWidth = trashSize[1]
            trashHeight = trashSize[0]
            trashX = padWidth
            trashY = random.randrange(0, padHeight - trashHeight)
            trashPassed += 1
            if trashPassed < 3:         # 쓰레기 2개까지 사운드 추가
                trashMissSound.play()

        if trashPassed == 3:            # 쓰레기 3개 놓치면 게임오버
            gameOver2()

        writeScore(score)


        # 놓친 쓰레기 수 표시
        writePassed(trashPassed)

        fishX -= fishSpeed             # 물고기 움직임
        fish2X -= fish2Speed 
        
        # 물고기가 왼쪽으로 이동한 경우
        if fishX <= 0:
            # 새로운 물고기 (랜덤)
            fish = pygame.image.load(random.choice(fishImage))
            fishSize = fish.get_rect().size
            fishWidth = fishSize[1]
            fishHeight = fishSize[0]
            fishX = padWidth
            fishY = random.randrange(0, padHeight - fishHeight)

        if fish2X <= 0: 
            # 새로운 물고기 (랜덤)
            fish2 = pygame.image.load(random.choice(fishImage2))
            fish2Size = fish2.get_rect().size
            fish2Width = fish2Size[1]
            fish2Height = fish2Size[0]
            fish2X = padWidth
            fish2Y = random.randrange(0, padHeight - fish2Height)

        sharkX -= sharkSpeed             # 상어 움직임
        
        # 상어가 왼쪽으로 이동한 경우
        if sharkX <= 0:
            # 새로운 상어 (랜덤)
            shark = pygame.image.load(random.choice(sharkImage))
            sharkSize = shark.get_rect().size
            sharkWidth = sharkSize[1]
            sharkHeight = sharkSize[1]
            sharkX = padWidth
            sharkY = random.randrange(0, padHeight - sharkHeight)

        shark2X -= shark2Speed             # 상어 움직임

        if shark2X <= 0:
            # 새로운 상어 (랜덤)
            shark2 = pygame.image.load(random.choice(shark2Image))
            shark2Size = shark2.get_rect().size
            shark2Width = shark2Size[1]
            shark2Height = shark2Size[1]
            shark2X = padWidth
            shark2Y = random.randrange(0, padHeight - shark2Height)

        # 쓰레기를 맞춘 경우
        if isShot:
            # 쓰레기 폭발
            drawObject(plus, trashX, trashY)       # 쓰레기 폭발 그리기
            destroySound.play()                    # 쓰레기 폭발 사운드 재생

            # 새로운 쓰레기 (랜덤)
            trash = pygame.image.load(random.choice(trashImage))
            trash = pygame.transform.rotate(trash,random.randrange(0,180))
            trashSize = trash.get_rect().size
            trashWidth = trashSize[0]
            trashHeight = trashSize[1]
            trashX = padWidth
            trashY = random.randrange(0, padHeight - trashHeight)
            destroySound = pygame.mixer.Sound(random.choice(trashSound))
            isShot = False

            # 쓰레기 맞추면 속도 증가
            trashSpeed += 0.02
            if trashSpeed >= 10:
                trashSpeed = 10

        # 물고기를 맞춘 경우
        if Shot:
            # 물고기 폭발
            drawObject(minus, fishX, fishY)         # 물고기 폭발 그리기
            CatchSound.play()                       # 물고기 폭발 사운드 재생


            # 새로운 물고기 (랜덤)
            fish = pygame.image.load(random.choice(fishImage))
            fishSize = fish.get_rect().size
            fishWidth = fishSize[0]
            fishHeight = fishSize[1]
            fishX = padWidth
            fishY = random.randrange(0, padHeight - fishHeight)
            CatchSound = pygame.mixer.Sound(random.choice(fishSound))
            Shot = False

        if Shot2: 
            # 물고기 폭발
            drawObject2(minus, fish2X, fish2Y)       # 물고기 폭발 그리기
            CatchSound.play()                        # 물고기 폭발 사운드 재생

            fish2 = pygame.image.load(random.choice(fishImage2)) 
            fish2Size = fish2.get_rect().size
            fish2Width = fish2Size[0]
            fish2Height = fish2Size[1]
            fish2X = padWidth
            fish2Y = random.randrange(0, padHeight - fish2Height)
            CatchSound = pygame.mixer.Sound(random.choice(fishSound))
            Shot2 = False

        drawObject(trash, trashX, trashY)      # 쓰레기 그리기

        drawObject(fish, fishX, fishY)         # 물고기 그리기

        drawObject2(fish2, fish2X, fish2Y)

        drawObject(shark, sharkX, sharkY)      # 상어 그리기

        drawObject2(shark2, shark2X, shark2Y)
        
        pygame.display.update()                # 게임화면을 다시그림

        clock.tick(60)                         # 게임화면의 초당 프레임수를 60으로 설정

    pygame.quit()         # pygame 종료

initGame()
start()
runGame()
runGame2()
