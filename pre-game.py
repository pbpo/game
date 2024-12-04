import pygame
import numpy as np
import random

# Pygame 초기화
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Crofton Length Game")
font = pygame.font.SysFont(None, 24)

# 게임 변수 초기화
drawing = False
points = []
tbox_w, tbox_h = 300, 100
total_players = 5  # 총 플레이어 수 (5명)
current_player = 0
player_lengths_pre = [None] * total_players  # 사전 게임에서 각 플레이어의 선 길이
player_surfaces_pre = [pygame.Surface(screen.get_size()) for _ in range(total_players)]
for surf in player_surfaces_pre:
    surf.fill((0, 0, 0))

# 사전 게임 변수
phase = 'pre_game'
target_number_pre = random.uniform(1000, 5000)  # 사전 게임의 랜덤 타겟 숫자
order = []  # 게임 진행 순서
target_number_main = random.uniform(1000, 5000)  # 본 게임의 랜덤 타겟 숫자
player_lengths_main = [None] * total_players  # 본 게임에서 각 플레이어의 선 길이
player_surfaces_main = [pygame.Surface(screen.get_size()) for _ in range(total_players)]
for surf in player_surfaces_main:
    surf.fill((0, 0, 0))

winner = None
min_diff = None

# FFT 평활화 함수
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
    # 저주파 성분 유지
    x_fft_filtered[:keep] = x_fft[:keep]
    x_fft_filtered[-keep:] = x_fft[-keep:]
    y_fft_filtered[:keep] = y_fft[:keep]
    y_fft_filtered[-keep:] = y_fft[-keep:]
    x_smooth = np.fft.ifft(x_fft_filtered).real
    y_smooth = np.fft.ifft(y_fft_filtered).real
    smoothed_points = list(zip(x_smooth, y_smooth))
    return smoothed_points

# 크로프턴 길이 계산 함수
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

def count_intersections(line):
    return np.count_nonzero(np.diff(line.astype(int)))

# 지침 표시 함수
def display_instructions():
    if phase == 'pre_game':
        instructions = [
            f"Pre-Game Phase: Target Number = {target_number_pre:.1f}",
            f"Player {current_player +1}: Draw when you're ready.",
            "Left Click: Start/Continue Drawing",
            "Right Click: Clear Screen",
        ]
    elif phase == 'game':
        instructions = [
            f"Main Game Phase: Target Number = {target_number_main:.1f}",
            f"Current Player: Player {order[current_player] +1}",
            "Left Click: Start/Continue Drawing",
            "Right Click: Clear Screen",
            "",
            "Player Order:"
        ]
        # 순서 표시
        for idx, player in enumerate(order):
            order_text = f"{idx +1}. Player {player +1}"
            instructions.append(order_text)
    elif phase == 'result':
        instructions = [
            "Game Over!",
            f"Winner: Player {winner +1} (Difference: {min_diff:.1f})",
            "Press Close to Exit."
        ]
    for idx, text in enumerate(instructions):
        rendered_text = font.render(text, True, (255, 255, 255))
        screen.blit(rendered_text, (10, 10 + idx * 20))

# 메인 함수
def main():
    global drawing, points, current_player, phase, order, target_number_pre, winner, min_diff
    clock = pygame.time.Clock()
    play = True

    while play:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                play = False
            elif phase in ['pre_game', 'game']:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        drawing = True
                        points = []
                    elif event.button == 3:
                        # 현재 플레이어의 화면을 초기화
                        if phase == 'pre_game':
                            surf = player_surfaces_pre[current_player]
                        elif phase == 'game':
                            surf = player_surfaces_main[order[current_player]]
                        surf.fill((0, 0, 0))
                        points = []
                        screen.blit(surf, (0, 0))
                        pygame.display.flip()
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button ==1 and drawing:
                        drawing = False
                        if len(points) > 2:
                            smoothed_points = fft_smooth(points)
                            if phase == 'pre_game':
                                surf = player_surfaces_pre[current_player]
                            elif phase == 'game':
                                surf = player_surfaces_main[order[current_player]]
                            surf.fill((0, 0, 0))

                            pygame.draw.lines(surf, (255, 255, 255), False, smoothed_points, 3)
                            crofton_curve_length = crofton_length(surf)

                            if phase == 'pre_game':
                                player_lengths_pre[current_player] = crofton_curve_length
                                # 화면에 표시
                                screen.blit(surf, (0, 0))
                                text = font.render(f'Player {current_player +1} Length: {crofton_curve_length:.1f}', True, (255, 255, 255))
                            elif phase == 'game':
                                player_lengths_main[order[current_player]] = crofton_curve_length
                                # 화면에 표시
                                screen.blit(surf, (0, 0))
                                text = font.render(f'Player {order[current_player] +1} Length: {crofton_curve_length:.1f}', True, (255, 255, 255))
                            screen.blit(text, (10, tbox_h + (current_player if phase == 'pre_game' else current_player)* 20))
                            pygame.display.flip()

                            if phase == 'pre_game':
                                current_player +=1
                                if current_player >= total_players:
                                    # 사전 게임 종료: 순서 결정
                                    differences = [abs(length - target_number_pre) for length in player_lengths_pre]
                                    # 차이 기준으로 플레이어 정렬
                                    sorted_players = sorted(range(total_players), key=lambda i: differences[i])
                                    # 가장 가까운 플레이어를 마지막으로 이동
                                    order = sorted_players.copy()
                                    closest_player = order.pop(0)  # 가장 작은 차이를 가진 플레이어
                                    order.append(closest_player)
                                    # 게임 단계 전환
                                    phase = 'game'
                                    current_player =0
                                    # 화면 초기화
                                    screen.fill((0,0,0))
                                    pygame.display.flip()
                            elif phase == 'game':
                                current_player +=1
                                if current_player >= total_players:
                                    # 본 게임 종료: 승자 결정
                                    differences_main = [abs(length - target_number_main) for length in player_lengths_main]
                                    min_diff = min(differences_main)
                                    winner = differences_main.index(min_diff)
                                    phase = 'result'
                                    screen.fill((0, 0, 0))
                                    result_text = font.render(f'Winner: Player {winner +1} (Difference: {min_diff:.1f})', True, (255, 255, 0))
                                    screen.blit(result_text, (10, 10))
                                    pygame.display.flip()
                elif event.type == pygame.MOUSEMOTION and drawing:
                    x, y = event.pos
                    points.append((x, y))
                    if phase == 'pre_game':
                        surf = player_surfaces_pre[current_player]
                    elif phase == 'game':
                        surf = player_surfaces_main[order[current_player]]
                    if len(points) >1:
                        pygame.draw.line(surf, (255, 255, 255), points[-2], points[-1], 3)
                        screen.blit(surf, (0, 0))
                    pygame.display.flip()

        # 사용자 인터페이스 업데이트
        if phase != 'result':
            # 지침 표시 영역 초기화
            pygame.draw.rect(screen, (0, 0, 0), (0,0, tbox_w, tbox_h))
            display_instructions()
            pygame.display.flip()

        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
