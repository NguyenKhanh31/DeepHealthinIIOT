import pygame, sys, random
#tao ham
def draw_flr():
	screen.blit(flr,(flr_x_pos,650))
	screen.blit(flr,(flr_x_pos+500,650))
def create_pipe():
	random_pipe_pos = random.choice(pipe_height)
	bottom_pipe = pipe_surface.get_rect(midtop =(500,random_pipe_pos))
	top_pipe = pipe_surface.get_rect(midtop =(500,random_pipe_pos- 750))
	return	bottom_pipe, top_pipe
def move_pipe(pipes):
	for pipe in pipes :
		pipe.centerx -= 5
	return pipes	
def draw_pipe(pipes):
	for pipe in pipes:
		if pipe.bottom >= 600:
			screen.blit(pipe_surface,pipe)	
		else:
			flip_pipe = pygame.transform.flip(pipe_surface,False,True)	
			screen.blit(flip_pipe,pipe)
def check_collision(pipes):
	for pipe in pipes:
		if bird_rect.colliderect(pipe):
			return False
	if bird_rect.top <= -75 or bird_rect.bottom >= 650:
		return False
	return True 	
def rotate_bird(bird1):
	new_bird = pygame.transform.rotozoom(bird1,-bird_movement*3,1)
	return new_bird
def score_display(game_state):
	if game_state == 'main game':
		score_surface = game_font.render(str(int(score)),True,(255,255,255))
		score_rect = score_surface.get_rect(center = (216,100))
		screen.blit(score_surface,score_rect)
	if game_state == 'game_over':	
		score_surface = game_font.render(f'Score: {int(score)}',True,(255,255,255))
		score_rect = score_surface.get_rect(center = (216,100))
		screen.blit(score_surface,score_rect)

		high_score_surface = game_font.render(f'High score: {int(high_score)}',True,(255,255,255))
		high_score_rect = high_score_surface.get_rect(center = (216,630))
		screen.blit(high_score_surface,high_score_rect)
def update_score(score,high_score):
	if score > high_score:
		high_score = score
	return 	high_score		
pygame.init()
screen= pygame.display.set_mode((432,768))
clock = pygame.time.Clock()
game_font = pygame.font.Font('04B_19.ttf',40)
#tao bien
gravity = 0.25
bird_movement = 0
game_active = True 
score = 0
high_score = 0
#background
bg = pygame.image.load('background-night.png')
bg = pygame.transform.scale2x(bg)
#san
flr = pygame.image.load('floor.png')
flr = pygame.transform.scale2x(flr)
flr_x_pos = 0
#chim
bird = pygame.image.load('mat (1).png').convert_alpha()
bird = pygame.transform.scale2x(bird)
bird_rect = bird.get_rect(center = (100,384))
#ong
pipe_surface = pygame.image.load('pipe-green.png')
pipe_surface = pygame.transform.scale2x(pipe_surface)
pipe_list = []
#timer
spawnpipe = pygame.USEREVENT
pygame.time.set_timer(spawnpipe, 1200)
pipe_height = [200,300,400]
#man hinh hoan thanh
game_over_surface = pygame.image.load('message.png').convert_alpha()
game_over_surface = pygame.transform.scale2x(game_over_surface)
game_over_rect = game_over_surface.get_rect(center=(216,384))
while True:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()
			sys.exit()
		if 	event.type == pygame.KEYDOWN:
			if event.key == pygame.K_SPACE and game_active:
				bird_movement = 0
				bird_movement =-7
			if event.key == pygame.K_SPACE and game_active == False:
				game_active = True
				pipe_list.clear()
				bird_rect.center = (100,384)
				bird_movement = 0
				score = 0 	
		if event.type == spawnpipe:
			pipe_list.extend(create_pipe())	

	screen.blit(bg,(0,0))
	if game_active:
		#chim di chuyen 
		bird_movement += gravity
		rotated_bird = rotate_bird(bird)
		bird_rect.centery += bird_movement
		screen.blit(rotated_bird,bird_rect)
		game_active = check_collision(pipe_list)
		#ong di chuyen
		pipe_list = move_pipe(pipe_list)
		draw_pipe(pipe_list)
		score += 0.01
		score_display('main game')
	else:
		screen.blit(game_over_surface,game_over_rect)
		high_score = update_score(score,high_score)
		score_display('game_over')	
	#san di chuyen
	flr_x_pos -= 1
	draw_flr()
	if flr_x_pos <= -432:
		flr_x_pos =0
	pygame.display.update()		
	clock.tick(80)