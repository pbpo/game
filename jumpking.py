import pygame
import random
import math
import numpy as np

# Pygame Initialization
pygame.init()
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))

pygame.display.set_caption("Star Catch Game")
font = pygame.font.SysFont(None, 24)

# Define Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
GRAY = (100, 100, 100)

# Game Variables Initialization
clock = pygame.time.Clock()
FPS = 60

# Define Stages (Length starts at 1000 and decreases by 100 each stage)
stages = [
    {"length": 100000, "jumps": 5},
    {"length": 90000, "jumps": 4},
    {"length": 80000, "jumps": 3},
    {"length": 70000, "jumps": 2},
]
current_stage_index = 0

# Game Phase
phase = "draw"  # Start directly with the "draw" phase

# Drawing Variables
drawing = False
points = []
total_drawn_length = 0
max_length = stages[current_stage_index]["length"]
platforms = []  # Each platform is a tuple of two points [(x1, y1), (x2, y2)]
exceeded_limit = False  # 한도 초과 여부

# Character Variables
character_radius = 20
character_color = BLUE
ground_height = 20  # Height of the ground
character_x, character_y = 50, screen_height - ground_height - character_radius
character_vx, character_vy = 0, 0
base_gravity = 0.5
gravity = base_gravity
jump_strength = -10
remaining_jumps = stages[current_stage_index]["jumps"]
on_ground = False
move_speed = 5  # Horizontal movement speed

# Gravity Increment per Stage
gravity_increment = 0.05  # Gravity increases by this amount each stage

# Star Variables
star_radius = 15
star_position = (
    screen_width - 100,
    random.randint(100, screen_height - ground_height - star_radius - 100),
)

# UI Display Positions
length_display_pos = (10, 10)
jumps_display_pos = (10, 40)
stage_display_pos = (10, 70)

def intersections(arr):
    count = 0
    for i in range(1, len(arr)):
        if arr[i] != arr[i-1]:
            count += 1
    return count

def crofton_length(screen, step=10):
    img = pygame.surfarray.pixels3d(screen)
    non = np.zeros((screen_height, screen_width), dtype=np.uint8)
    
    # 흰색 픽셀 검출
    for i in range(screen_height):
        for j in range(screen_width):
            non[i, j] = 1 if (img[j, i][0] == 255 and img[j, i][1] == 255 and img[j, i][2] == 255) else 0
    
    del img
    
    total_lines = (screen_width // step) + (screen_height // step)
    pixel_size = step  # 픽셀 크기를 step으로 설정
    
    num_intrs = 0
    # 수직선과의 교차점
    for x in range(0, screen_width, step):
        column = non[:, x]
        crossings = np.sum(np.abs(np.diff(column)))
        num_intrs += crossings
    
    # 수평선과의 교차점
    for y in range(0, screen_height, step):
        row = non[y, :]
        crossings = np.sum(np.abs(np.diff(row)))
        num_intrs += crossings
    
    # Crofton 공식 적용
    # L = (π/2) * (h/n) * I
    # h: 전체 길이(화면 width + height)
    # n: 그리드 라인 수
    # I: 총 교차점 수
    total_length = (screen_width + screen_height)
    length = (math.pi / 2) * (total_length / total_lines) * (num_intrs / 2)
    
    return length

def display_text(text, pos, color=WHITE):
    rendered_text = font.render(text, True, color)
    screen.blit(rendered_text, pos)

def check_collision(cx, cy, radius, p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    l2 = dx*dx + dy*dy
    if l2 == 0:
        d2 = (cx - p1[0])**2 + (cy - p1[1])**2
        return d2 <= radius*radius, p1[0], p1[1]
    
    t = max(0, min(1, ((cx - p1[0])*dx + (cy - p1[1])*dy)/l2))
    proj_x = p1[0] + t*dx
    proj_y = p1[1] + t*dy

    cross_product = abs((cx - p1[0]) * dy - (cy - p1[1])*dx)
    distance = cross_product / math.sqrt(l2)
    
    if distance <= radius:
        return True, proj_x, proj_y
    return False, None, None

def check_collision_krafton(char_x, char_y, platforms, vy):
    global character_y, character_vy, on_ground, remaining_jumps
    
    for platform in platforms:
        p1, p2 = platform
        collided, closest_x, closest_y = check_collision(
            char_x, char_y, character_radius, p1, p2
        )
        
        if collided:
            if vy > 0:
                if p2[0] - p1[0] != 0:
                    slope = (p2[1] - p1[1])/(p2[0]-p1[0])
                    if min(p1[0], p2[0]) <= char_x <= max(p1[0], p2[0]):
                        platform_y = slope * (char_x - p1[0]) + p1[1]
                        character_y = platform_y - character_radius
                        character_vy = 0
                        on_ground = True
                        # remaining_jumps = stages[current_stage_index]["jumps"]  # 이 줄 제거
                        return
                else:
                    if min(p1[1], p2[1]) <= char_y <= max(p1[1], p2[1]):
                        character_x = p1[0] - character_radius
                        character_vy = 0
    on_ground = False

def reset_stage():
    global drawing, points, total_drawn_length, platforms
    global character_x, character_y, character_vx, character_vy, on_ground
    global remaining_jumps, star_position
    global gravity, max_length, exceeded_limit

    drawing = False
    points = []
    total_drawn_length = 0
    platforms = []
    exceeded_limit = False

    ground_platform = [
        (0, screen_height - ground_height),
        (screen_width, screen_height - ground_height),
    ]
    platforms.append(ground_platform)

    character_x, character_y = 50, screen_height - ground_height - character_radius
    character_vx, character_vy = 0, 0
    on_ground = False

    remaining_jumps = stages[current_stage_index]["jumps"]
    gravity = base_gravity + gravity_increment * current_stage_index

    # 별 위치 재설정 시 캐릭터와 일정 거리 이상 떨어지도록 조정
    while True:
        sx = random.randint(screen_width // 2, screen_width - 100)
        sy = random.randint(100, screen_height - ground_height - star_radius - 100)
        dist = math.hypot(sx - character_x, sy - character_y)
        if dist > 200:
            star_position = (sx, sy)
            break

    max_length = stages[current_stage_index]["length"]

def move_character():
    global character_x, character_y, character_vx, character_vy, on_ground, remaining_jumps

    character_vy += gravity
    character_y += character_vy
    character_x += character_vx

    if character_x - character_radius < 0:
        character_x = character_radius
    elif character_x + character_radius > screen_width:
        character_x = screen_width - character_radius

    if character_y + character_radius > screen_height - ground_height:
        character_y = screen_height - ground_height - character_radius
        character_vy = 0
        on_ground = True
        # remaining_jumps = stages[current_stage_index]["jumps"]  # 이 줄 제거
    else:
        on_ground = False

    check_collision_krafton(character_x, character_y, platforms, character_vy)

def jump():
    global character_vy, remaining_jumps, on_ground
    if remaining_jumps > 0:
        character_vy = jump_strength
        on_ground = False
    remaining_jumps = max(remaining_jumps - 1, 0)

def draw_platforms():
    for platform in platforms:
        pygame.draw.line(screen, GREEN, platform[0], platform[1], 4)

def draw_star():
    pygame.draw.circle(screen, YELLOW, star_position, star_radius)

def check_star_reached():
    distance = math.hypot(character_x - star_position[0], character_y - star_position[1])
    return distance < (character_radius + star_radius)

def next_stage():
    global current_stage_index, phase
    current_stage_index += 1
    if current_stage_index >= len(stages):
        phase = "result"
    else:
        phase = "draw"
        reset_stage()

# Initialize the first stage
reset_stage()

running = True
while running:
    clock.tick(FPS)
    screen.fill(BLACK)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif phase == "draw":
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    drawing = True
                    points = [event.pos]
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and drawing:
                    drawing = False
                    if len(points) > 1:
                        temp_surface = screen.copy()
                        pygame.draw.aalines(temp_surface, WHITE, False, points, 2)
                        
                        total_length_test = crofton_length(temp_surface)

                        if total_length_test <= max_length:
                            pygame.draw.aalines(screen, WHITE, False, points, 2)
    # 연속된 두 점씩 플랫폼으로 저장
                            for i in range(len(points)-1):
                                platforms.append((points[i], points[i+1]))
                            total_drawn_length = total_length_test
                            phase = "play"
                        else:
                            exceeded_limit = True
                            points = []
            elif event.type == pygame.MOUSEMOTION:  # 이 부분을 추가
                if drawing:
                    points.append(event.pos)  # 마우스 이동 시 점 추가

        elif phase == "play":
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    jump()

        elif phase == "result":
            pass

    keys = pygame.key.get_pressed()
    if phase == "play":
        if keys[pygame.K_LEFT]:
            character_vx = -move_speed
        elif keys[pygame.K_RIGHT]:
            character_vx = move_speed
        else:
            character_vx = 0

    if phase == "draw":
        # 현재까지 그린 플랫폼 다시 그리기
        draw_platforms()
        draw_star()
        pygame.draw.circle(screen, character_color, (int(character_x), int(character_y)), character_radius)
        display_text(f"Stage {current_stage_index +1}: Draw up to {max_length} units (Crofton approx)", stage_display_pos)
        display_text(f"Total Drawn Length (Crofton): {total_drawn_length:.1f} / {max_length}", length_display_pos)
        display_text("Left Click and Drag to Draw (Release to Finish)", (10, 100))
        display_text("Create Curved Lines by Drawing Smoothly", (10, 130))
        display_text("You must stay within the length limit!", (10, 160))

        # 한도 초과 안내
        if exceeded_limit:
            display_text("Length limit exceeded! Try a shorter line.", (10, 190), RED)

        # 마우스 드래그 시 실시간으로 곡선(부드러운 폴리라인) 그리기
        if drawing and len(points) > 1:
            pygame.draw.aalines(screen, WHITE, False, points, 2)

    elif phase == "play":
        move_character()

        pygame.draw.circle(screen, character_color, (int(character_x), int(character_y)), character_radius)
        draw_platforms()
        draw_star()

        display_text(f"Stage {current_stage_index +1}: Play Phase", stage_display_pos)
        display_text(f"Remaining Jumps: {remaining_jumps}", jumps_display_pos)
        remaining_length_text = max_length - total_drawn_length
        display_text(f"Remaining Length (approx): {remaining_length_text:.1f}", length_display_pos)

        if check_star_reached():
            display_text("You Reached the Star! Press Enter to Continue.", (10, 100), YELLOW)
            phase = "result"

    elif phase == "result":
        if current_stage_index < len(stages):
            display_text("Stage Cleared!", (10, 10), GREEN)
            display_text("Press Enter to Proceed to Next Stage.", (10, 40), GREEN)
        else:
            display_text("All Stages Completed! You Win!", (10, 10), YELLOW)
            display_text("Press Enter to Exit.", (10, 40), YELLOW)

        if keys[pygame.K_RETURN]:
            if current_stage_index < len(stages):
                next_stage()
            else:
                running = False

    # 점프 소진 체크를 제거하고, 실패 조건을 다음과 같이 변경
    if phase == "play":
    # 화면 하단을 벗어났을 때만 실패 처리
        if character_y + character_radius > screen_height:
            display_text("Failed! Press Enter to Retry.", (10, 100), RED)
            phase = "result"
            if keys[pygame.K_RETURN]:
                reset_stage()
                phase = "draw"
    pygame.display.flip()

pygame.quit()

