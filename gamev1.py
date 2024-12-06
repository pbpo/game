import pygame
import numpy as np
import random
import time

pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Crofton Length Game - Enhanced")
font = pygame.font.SysFont(None, 24)

player_colors = [
    (255,255,255),
    (255,0,0),
    (0,255,0),
    (0,0,255),
    (255,255,0)
]

line_thickness = 3
max_thickness = 10
min_thickness = 1

pygame.mixer.init()
try:
    pygame.mixer.music.load("background_music.mp3")  
    pygame.mixer.music.play(-1)
except:
    pass

try:
    success_sound = pygame.mixer.Sound("success.wav")
    fail_sound = pygame.mixer.Sound("fail.wav")
    hint_sound = pygame.mixer.Sound("hint.wav")
    elimination_sound = pygame.mixer.Sound("elimination.wav")
except:
    success_sound = None
    fail_sound = None
    hint_sound = None
    elimination_sound = None

total_players = 5
active_players = list(range(total_players))
round_count = total_players -1
current_round = 1

difficulty = 'Normal'
if difficulty == 'Easy':
    target_range = (1000, 3000)
    time_limit = 20
elif difficulty == 'Hard':
    target_range = (3000, 6000)
    time_limit = 10
else:
    target_range = (1000, 5000)
    time_limit = 15

achievement_count = 0
achievement_unlocked = False

story_text = "You are in a mystical forest, seeking to please a guardian spirit.\n" \
             "The spirit demands a line of light drawn to a specific length.\n" \
             "Whoever gets closest to the spirit's desired length shall be blessed!"

player_lengths_pre = [[None]*total_players for _ in range(round_count)]
player_lengths_main = [[None]*total_players for _ in range(round_count)]

player_surfaces_pre = [[pygame.Surface(screen.get_size()) for _ in range(total_players)] for _ in range(round_count)]
player_surfaces_main = [[pygame.Surface(screen.get_size()) for _ in range(total_players)] for _ in range(round_count)]

for r in range(round_count):
    for surf in player_surfaces_pre[r]:
        surf.fill((0, 0, 0))
    for surf in player_surfaces_main[r]:
        surf.fill((0, 0, 0))

phase = 'pre_game'
drawing = False
points = []
current_player_index = 0
winner = None
min_diff = None

target_numbers_pre = [random.uniform(*target_range) for _ in range(round_count)]
target_numbers_main = [random.uniform(*target_range) for _ in range(round_count)]
order = active_players.copy()

start_time = None

hint_message = None
hint_start_time = None
HINT_DURATION = 2.0

eliminated_players = []

def fft_smooth(points, keep_fraction=0.05):
    x_coords = np.array([p[0] for p in points])
    y_coords = np.array([p[1] for p in points])
    n = len(points)
    if n == 0:
        return []
    x_fft = np.fft.fft(x_coords)
    y_fft = np.fft.fft(y_coords)
    keep = int(n * keep_fraction / 2)
    if keep < 1:
        keep = 1
    x_fft_filtered = np.zeros_like(x_fft)
    y_fft_filtered = np.zeros_like(y_fft)
    x_fft_filtered[:keep] = x_fft[:keep]
    x_fft_filtered[-keep:] = x_fft[-keep:]
    y_fft_filtered[:keep] = y_fft[:keep]
    y_fft_filtered[-keep:] = y_fft[-keep:]
    x_smooth = np.fft.ifft(x_fft_filtered).real
    y_smooth = np.fft.ifft(y_fft_filtered).real
    smoothed_points = list(zip(x_smooth, y_smooth))
    return smoothed_points

def count_intersections(line):
    return np.count_nonzero(np.diff(line.astype(int)))

def crofton_length(surface, step=10):
    img = pygame.surfarray.array3d(surface).transpose([1, 0, 2])
    height, width, _ = img.shape
    non_zero = np.any(img != 0, axis=2)
    intersections_count = 0

    for x in range(0, width, step):
        col = non_zero[:, x]
        intersections_count += count_intersections(col)

    for y in range(0, height, step):
        row = non_zero[y, :]
        intersections_count += count_intersections(row)

    estimated_length = (np.pi / 2) * intersections_count * step
    return estimated_length

def display_instructions():
    instructions = []
    instructions.append(f"Difficulty: {difficulty}")
    instructions.append(f"Round: {current_round}/{round_count}")
    instructions.append(f"Active Players: {len(active_players)}")
    if phase == 'pre_game':
        instructions.append(f"Pre-Game Phase: Target = {target_numbers_pre[current_round-1]:.1f}")
        instructions.append(f"Player {active_players[current_player_index]+1}'s turn (Time limit: {time_limit}s)")
        instructions.append("Left Click: Start/Continue Drawing   Right Click: Clear")
        instructions.append("Fixed Colors: Each player has their own color")
    elif phase == 'game':
        instructions.append(f"Main Game Phase: Target = {target_numbers_main[current_round-1]:.1f}")
        instructions.append(f"Current Player: Player {order[current_player_index]+1} (Time limit: {time_limit}s)")
        instructions.append("Left Click: Start/Continue Drawing   Right Click: Clear")
        instructions.append("Fixed Colors: Each player has their own color")
    elif phase == 'elimination':
        instructions.append(f"Player {eliminated_players[-1]+1} has been eliminated!")
    elif phase == 'result':
        instructions.append("Game Over!")
        instructions.append(f"Winner: Player {winner+1} (Difference: {min_diff:.1f})")
        instructions.append("Close the window to exit.")

    if achievement_unlocked:
        instructions.append("Achievement Unlocked! Close match (within 100) at least 3 times!")

    for i, text in enumerate(instructions):
        rendered = font.render(text, True, (255,255,255))
        screen.blit(rendered, (10, 10+i*20))

def display_story():
    lines = story_text.split('\n')
    x = 500
    y = 10
    for line in lines:
        rendered = font.render(line, True, (200,200,200))
        screen.blit(rendered, (x,y))
        y += 20

def display_hints(current_length, target_number):
    global hint_message, hint_start_time
    
    diff = current_length - target_number
    hint_line = f"Current Length: {current_length:.1f}, Target: {target_number:.1f}, Diff: {abs(diff):.1f}"
    if diff > 0:
        hint_line += " (Too Long!)"
    else:
        hint_line += " (Too Short!)"
    
    hint_message = hint_line
    hint_start_time = time.time()

def draw_current_hint():
    global hint_message, hint_start_time
    
    if hint_message and hint_start_time:
        current_time = time.time()
        if current_time - hint_start_time < HINT_DURATION:
            rendered = font.render(hint_message, True, (255,255,0))
            screen.blit(rendered, (10, 300))
        else:
            hint_message = None
            hint_start_time = None

def finalize_round():
    global phase, current_round, order, winner, min_diff, achievement_unlocked
    differences = []
    for p in active_players:
        total_diff = 0
        for r in range(current_round-1):
            if player_lengths_main[r][p] is not None:
                total_diff += abs(player_lengths_main[r][p] - target_numbers_main[r])
        differences.append((p, total_diff))
    differences.sort(key=lambda x: x[1])
    eliminated_player, max_diff = differences[-1]
    eliminated_players.append(eliminated_player)
    active_players.remove(eliminated_player)
    if elimination_sound:
        elimination_sound.play()
    phase = 'elimination'
    pygame.display.flip()
    pygame.time.delay(2000)
    if len(active_players) == 1:
        winner = active_players[0]
        min_diff = differences[0][1]
        check_achievement()
        phase = 'result'

def check_achievement():
    global achievement_count, achievement_unlocked
    for r in range(current_round):
        for p in range(total_players):
            l = player_lengths_main[r][p]
            if l is not None:
                diff = abs(l - target_numbers_main[r])
                if diff <= 100:
                    achievement_count += 1
    if achievement_count >= 3:
        achievement_unlocked = True

def catmull_rom_spline(points, resolution=100):
    if len(points) < 4:
        return points
    spline_points = []
    for i in range(len(points)-3):
        p0 = points[i]
        p1 = points[i+1]
        p2 = points[i+2]
        p3 = points[i+3]
        for t in np.linspace(0,1,resolution):
            t2 = t*t
            t3 = t2*t
            x = 0.5 * ((2*p1[0]) + (-p0[0] + p2[0])*t + (2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2 + (-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3)
            y = 0.5 * ((2*p1[1]) + (-p0[1] + p2[1])*t + (2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2 + (-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
            spline_points.append((x,y))
    return spline_points

def draw_elimination_animation(player):
    snake_color = (0, 255, 0)
    snake_pos = [400, 300]
    player_color = player_colors[player]
    for i in range(50):
        pygame.draw.circle(screen, snake_color, snake_pos, 20)
        pygame.draw.circle(screen, player_color, (snake_pos[0] + i, snake_pos[1]), 10)
        pygame.display.flip()
        pygame.time.delay(30)
        screen.fill((0,0,0))
        display_instructions()
        display_story()
    screen.fill((0,0,0))
    display_instructions()
    display_story()
    pygame.display.flip()

def main():
    global drawing, points, current_player_index, phase, order, winner, min_diff
    global line_thickness, current_round, start_time, achievement_unlocked
    
    clock = pygame.time.Clock()
    play = True

    screen.fill((0,0,0))
    display_instructions()
    display_story()
    pygame.display.flip()

    while play:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                play = False

            if phase in ['pre_game', 'game']:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        drawing = True
                        points = []
                        if start_time is None:
                            start_time = time.time()
                    elif event.button == 3:
                        if phase == 'pre_game':
                            surf = player_surfaces_pre[current_round-1][active_players[current_player_index]]
                        else:
                            surf = player_surfaces_main[current_round-1][order[current_player_index]]
                        surf.fill((0, 0, 0))
                        points = []
                        screen.blit(surf, (0, 0))
                        pygame.display.flip()

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button ==1 and drawing:
                        drawing = False
                        if len(points) > 2:
                            curve_points= fft_smooth(points)
                            

                            if phase == 'pre_game':
                                surf = player_surfaces_pre[current_round-1][active_players[current_player_index]]
                                target_number = target_numbers_pre[current_round-1]
                                player_color = player_colors[active_players[current_player_index]]
                            else:
                                surf = player_surfaces_main[current_round-1][order[current_player_index]]
                                target_number = target_numbers_main[current_round-1]
                                player_color = player_colors[order[current_player_index]]

                            surf.fill((0, 0, 0))
                            if len(curve_points) > 1:
                                pygame.draw.lines(surf, player_color, False, curve_points, line_thickness)
                            crofton_curve_length = crofton_length(surf)

                            if phase == 'pre_game':
                                player_lengths_pre[current_round-1][active_players[current_player_index]] = crofton_curve_length
                            else:
                                player_lengths_main[current_round-1][order[current_player_index]] = crofton_curve_length

                            if success_sound:
                                success_sound.play()

                            screen.blit(surf, (0, 0))
                            text = font.render(
                                f'Player {active_players[current_player_index]+1} Length: {crofton_curve_length:.1f}', 
                                True, (255, 255, 255))
                            screen.blit(text, (10, 420+(current_player_index*20)))
                            pygame.display.flip()

                            current_player_index +=1
                            if phase == 'pre_game':
                                if current_player_index >= len(active_players):
                                    phase = 'game'
                                    current_player_index =0
                                    start_time = None
                                    screen.fill((0,0,0))
                                    pygame.display.flip()
                            else:
                                if current_player_index >= len(active_players):
                                    current_player_index =0
                                    finalize_round()
                                    if phase != 'result':
                                        start_time = None
                                        screen.fill((0,0,0))
                                        pygame.display.flip()
                                else:
                                    start_time = time.time()

                elif event.type == pygame.MOUSEMOTION and drawing:
                    x, y = event.pos
                    points.append((x, y))
                    if phase == 'pre_game':
                        surf = player_surfaces_pre[current_round-1][active_players[current_player_index]]
                        target_number = target_numbers_pre[current_round-1]
                        player_color = player_colors[active_players[current_player_index]]
                    else:
                        surf = player_surfaces_main[current_round-1][order[current_player_index]]
                        target_number = target_numbers_main[current_round-1]
                        player_color = player_colors[order[current_player_index]]

                    if len(points)>1:
                        pygame.draw.line(surf, player_color, points[-2], points[-1], line_thickness)
                        screen.blit(surf, (0,0))
                        approx_length = crofton_length(surf)
                        display_hints(approx_length, target_number)
                    pygame.display.flip()

        if phase in ['pre_game','game'] and start_time is not None:
            elapsed = time.time() - start_time
            if elapsed > time_limit:
                drawing = False
                if fail_sound:
                    fail_sound.play()
                current_player_index +=1
                if phase == 'pre_game':
                    if current_player_index >= len(active_players):
                        phase = 'game'
                        current_player_index =0
                        start_time = None
                        screen.fill((0,0,0))
                        pygame.display.flip()
                    else:
                        start_time = time.time()
                else:
                    if current_player_index >= len(active_players):
                        finalize_round()
                        if phase != 'result':
                            start_time = None
                            screen.fill((0,0,0))
                            pygame.display.flip()
                    else:
                        start_time = time.time()

        if phase == 'elimination' and len(eliminated_players) >0:
            draw_elimination_animation(eliminated_players[-1])
            phase = 'pre_game'
            current_round +=1
            if current_round > round_count:
                winner = active_players[0]
                min_diff = 0
                check_achievement()
                phase = 'result'

        if phase != 'result':
            pygame.draw.rect(screen, (0, 0, 0), (0,0,800,600))
            
            if phase == 'pre_game':
                if current_player_index < len(active_players):
                    current_surf = player_surfaces_pre[current_round-1][active_players[current_player_index]]
                else:
                    current_surf = player_surfaces_pre[current_round-1][-1]
            elif phase == 'game':
                if order and current_player_index < len(order):
                    current_surf = player_surfaces_main[current_round-1][order[current_player_index]]
                else:
                    current_surf = player_surfaces_main[current_round-1][order[-1]] if order else player_surfaces_main[current_round-1][0]
            else:
                current_surf = None

            if current_surf is not None:
                screen.blit(current_surf, (0, 0))
            
            display_instructions()
            display_story()
            draw_current_hint()

            pygame.display.flip()

        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()