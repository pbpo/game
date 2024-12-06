import pygame
import random
import math

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
    {"length": 1000, "jumps": 5},
    {"length": 900, "jumps": 4},
    {"length": 800, "jumps": 3},
    {"length": 700, "jumps": 2},
    {"length": 600, "jumps": 1},
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
# Position the star higher above the ground
star_position = (
    screen_width - 100,
    random.randint(100, screen_height - ground_height - star_radius - 100),
)

# UI Display Positions
length_display_pos = (10, 10)
jumps_display_pos = (10, 40)
stage_display_pos = (10, 70)
instruction_pos = (10, 100)

# Function Definitions

def calculate_length(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def display_text(text, pos, color=WHITE):
    """Render and display text on the screen."""
    rendered_text = font.render(text, True, color)
    screen.blit(rendered_text, pos)

def check_collision(cx, cy, radius, p1, p2):
    
    # 선분의 벡터 계산
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    # 선분의 길이의 제곱
    l2 = dx * dx + dy * dy
    
    if l2 == 0:
        # p1과 p2가 같은 점인 경우
        d2 = (cx - p1[0]) * (cx - p1[0]) + (cy - p1[1]) * (cy - p1[1])
        return d2 <= radius * radius, p1[0], p1[1]
    
    # 원의 중심에서 선분에 내린 수선의 발을 구하는 매개변수 t
    t = max(0, min(1, ((cx - p1[0]) * dx + (cy - p1[1]) * dy) / l2))
    
    # 수선의 발 좌표 계산
    proj_x = p1[0] + t * dx
    proj_y = p1[1] + t * dy
    
    # 크래프톤 공식: 원의 중심에서 수선의 발까지의 거리 계산
    # |AP x AB| / |AB| = h (수선의 길이)
    # 여기서 AP는 원의 중심에서 선분의 시작점까지의 벡터
    # AB는 선분의 벡터
    cross_product = abs((cx - p1[0]) * dy - (cy - p1[1]) * dx)
    distance = cross_product / math.sqrt(l2)
    
    # 충돌 여부 확인
    if distance <= radius:
        # 충돌이 발생한 경우, 충돌점 반환
        return True, proj_x, proj_y
    
    return False, None, None

def check_collision_krafton(char_x, char_y, platforms, vy):
    """크래프톤 공식을 사용한 캐릭터와 플랫폼 간의 충돌 체크"""
    global character_y, character_vy, on_ground, remaining_jumps
    
    for platform in platforms:
        p1, p2 = platform
        collided, closest_x, closest_y = check_collision(
            char_x, char_y, character_radius, p1, p2
        )
        
        if collided:
            if vy > 0:  # 캐릭터가 아래로 이동 중일 때만 처리
                # 플랫폼의 기울기 계산
                if p2[0] - p1[0] != 0:  # 수직선이 아닌 경우
                    slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
                    # 캐릭터의 x 위치가 플랫폼의 범위 내에 있는지 확인
                    if min(p1[0], p2[0]) <= char_x <= max(p1[0], p2[0]):
                        # 플랫폼 위의 정확한 y 좌표 계산
                        platform_y = slope * (char_x - p1[0]) + p1[1]
                        # 캐릭터를 플랫폼 위로 위치시킴
                        character_y = platform_y - character_radius
                        character_vy = 0
                        on_ground = True
                        remaining_jumps = stages[current_stage_index]["jumps"]
                        return
                else:  # 수직선인 경우
                    if min(p1[1], p2[1]) <= char_y <= max(p1[1], p2[1]):
                        character_x = p1[0] - character_radius
                        character_vy = 0
    
    on_ground = False

def reset_stage():
    """Reset the current stage."""
    global drawing, points, total_drawn_length, platforms
    global character_x, character_y, character_vx, character_vy, on_ground
    global remaining_jumps, star_position
    global gravity, max_length

    drawing = False
    points = []
    total_drawn_length = 0
    platforms = []

    # Add the ground platform at the bottom of the screen
    ground_platform = [
        (0, screen_height - ground_height),
        (screen_width, screen_height - ground_height),
    ]
    platforms.append(ground_platform)

    # Reset character position and velocity
    character_x, character_y = 50, screen_height - ground_height - character_radius
    character_vx, character_vy = 0, 0
    on_ground = False

    # Reset remaining jumps
    remaining_jumps = stages[current_stage_index]["jumps"]

    # Adjust gravity based on the current stage
    gravity = base_gravity + gravity_increment * current_stage_index

    # Randomize star position above the ground
    star_position = (
        random.randint(screen_width // 2, screen_width - 100),
        random.randint(100, screen_height - ground_height - star_radius - 100),
    )

    # Update maximum drawing length for the stage
    max_length = stages[current_stage_index]["length"]

def move_character():
    """Handle the character's physics and movement."""
    global character_x, character_y, character_vx, character_vy, on_ground, remaining_jumps

    # Apply gravity
    character_vy += gravity
    character_y += character_vy
    character_x += character_vx

    # Check screen boundaries
    if character_x - character_radius < 0:
        character_x = character_radius
    elif character_x + character_radius > screen_width:
        character_x = screen_width - character_radius

    # Check collision with the ground
    if character_y + character_radius > screen_height - ground_height:
        character_y = screen_height - ground_height - character_radius
        character_vy = 0
        on_ground = True
        remaining_jumps = stages[current_stage_index]["jumps"]
    else:
        on_ground = False

    # Check collision with platforms
    check_collision_krafton(character_x, character_y, platforms, character_vy)

def jump():
    """Make the character jump and decrement the jump count."""
    global character_vy, remaining_jumps, on_ground
    if remaining_jumps > 0:
        character_vy = jump_strength
        on_ground = False
    # Decrement jump count on every space bar press, ensuring it doesn't go below 0
    remaining_jumps = max(remaining_jumps - 1, 0)

def draw_platforms():
    """Draw all platforms on the screen."""
    for platform in platforms:
        pygame.draw.line(screen, GREEN, platform[0], platform[1], 4)

def draw_star():
    """Draw the star on the screen."""
    pygame.draw.circle(screen, YELLOW, star_position, star_radius)

def check_star_reached():
    """Check if the character has reached the star."""
    distance = math.hypot(character_x - star_position[0], character_y - star_position[1])
    return distance < (character_radius + star_radius)

def next_stage():
    """Proceed to the next stage or end the game if all stages are completed."""
    global current_stage_index, phase
    current_stage_index += 1
    if current_stage_index >= len(stages):
        # All stages completed
        phase = "result"
    else:
        # Start the next stage
        phase = "draw"
        reset_stage()

# Initialize the first stage
reset_stage()

# Main Game Loop
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
                        # Draw lines between the points
                        pygame.draw.lines(screen, WHITE, False, points, 2)
                        # Add the platform
                        platforms.append((points[0], points[-1]))
                        # Calculate total drawn length
                        for i in range(len(points) - 1):
                            total_drawn_length += calculate_length(points[i], points[i + 1])
                        # Check if the drawn length exceeds the maximum allowed
                        if total_drawn_length > max_length:
                            # Trim the last line segment if it exceeds the limit
                            overflow = total_drawn_length - max_length
                            last_point = points[-1]
                            second_last_point = points[-2]
                            line_length = calculate_length(second_last_point, last_point)
                            if line_length != 0:
                                ratio = (line_length - overflow) / line_length
                                new_x = second_last_point[0] + (last_point[0] - second_last_point[0]) * ratio
                                new_y = second_last_point[1] + (last_point[1] - second_last_point[1]) * ratio
                                new_point = (int(new_x), int(new_y))
                                platforms[-1] = (second_last_point, new_point)
                                pygame.draw.line(screen, WHITE, second_last_point, new_point, 2)
                                total_drawn_length = max_length
                            # Transition to the play phase
                            phase = "play"

        elif phase == "play":
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    jump()

        elif phase == "result":
            # Handle result phase events if needed
            pass

    # Handle continuous key presses for movement
    keys = pygame.key.get_pressed()
    if phase == "play":
        if keys[pygame.K_LEFT]:
            character_vx = -move_speed
        elif keys[pygame.K_RIGHT]:
            character_vx = move_speed
        else:
            character_vx = 0

    # Draw based on the current phase
    if phase == "draw":
        # Draw platforms and the star
        draw_platforms()
        draw_star()
        # Draw the character
        pygame.draw.circle(screen, character_color, (int(character_x), int(character_y)), character_radius)
        # Display UI texts
        display_text(f"Stage {current_stage_index +1}: Draw up to {max_length} units", stage_display_pos)
        display_text(f"Total Drawn Length: {total_drawn_length:.1f} / {max_length}", length_display_pos)
        display_text("Left Click and Drag to Draw", (10, 100))
        display_text("Release Left Click to Finish Drawing", (10, 130))
        display_text("Draw Platforms to Reach the Star!", (10, 160))

    elif phase == "play":
        # Move the character
        move_character()

        # Draw the character
        pygame.draw.circle(screen, character_color, (int(character_x), int(character_y)), character_radius)

        # Draw platforms and the star
        draw_platforms()
        draw_star()

        # Display UI texts
        display_text(f"Stage {current_stage_index +1}: Play Phase", stage_display_pos)
        display_text(f"Remaining Jumps: {remaining_jumps}", jumps_display_pos)
        remaining_length_text = max_length - total_drawn_length
        display_text(f"Remaining Length: {remaining_length_text:.1f}", length_display_pos)

        # Check if the character has reached the star
        if check_star_reached():
            display_text("You Reached the Star! Press Enter to Continue.", (10, 100), YELLOW)
            phase = "result"

    elif phase == "result":
        # Display results
        if current_stage_index < len(stages):
            # Stage cleared
            display_text("Stage Cleared!", (10, 10), GREEN)
            display_text("Press Enter to Proceed to Next Stage.", (10, 40), GREEN)
        else:
            # All stages completed
            display_text("All Stages Completed! You Win!", (10, 10), YELLOW)
            display_text("Press Enter to Exit.", (10, 40), YELLOW)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_RETURN]:
            if current_stage_index < len(stages):
                next_stage()
            else:
                running = False

    # Check if the character is out of jumps and not on the ground
    if phase == "play" and remaining_jumps <= 0 and not on_ground:
        display_text("Out of Jumps! Press Enter to Retry.", (10, 100), RED)
        phase = "result"
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RETURN]:
            reset_stage()
            phase = "draw"

    # Handle real-time drawing
    if phase == "draw" and drawing:
        mouse_pos = pygame.mouse.get_pos()
        if len(points) > 0:
            pygame.draw.line(screen, WHITE, points[-1], mouse_pos, 2)
            current_length = calculate_length(points[-1], mouse_pos)
            if total_drawn_length + current_length <= max_length:
                points.append(mouse_pos)
                total_drawn_length += current_length
            else:
                # Draw only the remaining length to reach max_length
                remaining_length = max_length - total_drawn_length
                angle = math.atan2(mouse_pos[1] - points[-1][1], mouse_pos[0] - points[-1][0])
                new_x = points[-1][0] + remaining_length * math.cos(angle)
                new_y = points[-1][1] + remaining_length * math.sin(angle)
                new_point = (int(new_x), int(new_y))
                pygame.draw.line(screen, WHITE, points[-1], new_point, 2)
                platforms.append((points[-1], new_point))
                total_drawn_length = max_length
                drawing = False
                phase = "play"

    pygame.display.flip()

pygame.quit()
