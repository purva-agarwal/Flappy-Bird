import os, time, pygame, neat, random
pygame.font.init()

Window_width = 500
Window_height = 760

Gen = 0

Bird_images = [pygame.transform.scale2x(pygame.image.load (os.path.join("imgs", "bird1.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird2.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird3.png")))]
Pipe_image = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")))
Ground_image = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))
Background_image = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))

STAT_FONT = pygame.font.SysFont("monospace", 25)

class Bird:
	Birds = Bird_images
	Max_Rotation = 25
	Rotation_Veocity = 20
	Animation_Time = 5

	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.tilt = 0
		self.count = 0
		self.velocity = 0
		self.height = self.y
		self.image_count = 0
		self.images = self.Birds[0]
		

	def jump(self):
		self.velocity = -10.5
		self.count = 0
		self.height = self.y

	def move(self):
		self.count += 1
		d = self.velocity * self.count + 1.5 * self.count**2

		if d >= 16:
			d = 16

		if d < 0:
			d -=2

		self.y = self.y + d

		if d < 0 or self.y < self.height + 50:
			if self.tilt < self.Max_Rotation:
				self.tilt = self.Max_Rotation
		else:
			if self.tilt > -90:
				self.tilt -= self.Rotation_Veocity

	def draw(self, win):
		self.image_count +=1
		if self.image_count < self.Animation_Time:
			self.images = self.Birds[0]
		elif self.image_count < self.Animation_Time * 2:
			self.image = self.Birds[1]
		elif self.image_count < self.Animation_Time * 3:
			self.images = self.Birds[2]
		elif self.image_count < self.Animation_Time * 4:
			self.images = self.Birds[1]
		elif self.image_count == self.Animation_Time * 4 + 1:
			self.images = self.Birds[0]
			self.image_count = 0
		if self.tilt <= -80:
			self.images = self.Birds[1]
			self.image_count = self.Animation_Time * 2

		rotated_image = pygame.transform.rotate(self.images, self.tilt)
		rect = rotated_image.get_rect(center = self.images.get_rect(topleft = (self.x, self.y)).center)
		win.blit(rotated_image, rect.topleft)

	def get_mask(self):
		return pygame.mask.from_surface(self.images)

class Pipe:
	Gap = 200
	Velocity = 5

	def __init__(self, x):
		self.x = x
		self.height = 0

		self.top = 0
		self.bottom = 0
		self.Pipe_Top = pygame.transform.flip(Pipe_image, False, True)
		self.Pipe_Bottom = Pipe_image

		self.passed = False
		self.set_height()

	def set_height(self):
		self.height = random.randrange(50, 450)
		self.top = self.height - self.Pipe_Top.get_height()
		self.bottom = self.height + self.Gap

	def move(self):
		self.x -= self.Velocity 

	def draw(self, win):
		win.blit(self.Pipe_Top, (self.x, self.top))
		win.blit(self.Pipe_Bottom, (self.x, self.bottom))

	def collide(self, bird):
		bird_mask = bird.get_mask()
		
		top_mask = pygame.mask.from_surface(self.Pipe_Top)
		bottom_mask = pygame.mask.from_surface(self.Pipe_Bottom)
		
		top_offset =  (self.x - bird.x, self.top - int(round(bird.y)))
		bottom_offset = (self.x - bird.x, self.bottom - int(round(bird.y)))
		
		bottom_poc = bird_mask.overlap(bottom_mask, bottom_offset)
		top_poc = bird_mask.overlap(top_mask, top_offset)

		if top_poc or bottom_poc:
			return True

		return False

class Ground:
	Velocity = 5
	Width = Ground_image.get_width()
	G_Image = Ground_image

	def __init__(self, y):
		self.y = y
		self.x1 = 0
		self.x2 = self.Width

	def move(self):
		self.x1 -= self.Velocity 
		self.x2 -= self.Velocity

		if self.x1 + self.Width < 0:
			self.x1 = self.x2 + self.Width

		if self.x2 + self.Width < 0:
			self.x2 = self.x1 + self.Width

	def draw(self, win):
		win.blit(self.G_Image, (self.x1, self.y))
		win.blit(self.G_Image, (self.x2, self.y))

def draw_window(win, birds, pipes, ground, score, gen):
	win.blit(Background_image, (0,0))
	
	for pipe in pipes:
		pipe.draw(win)

	text = STAT_FONT.render("Score : " + str(score), 1, (255, 255, 255))
	win.blit(text, (Window_width - 10 - text.get_width(), 10))

	text = STAT_FONT.render("Generation : " + str(gen), 1, (255, 255, 255))
	win.blit(text, (10, 10))

	ground.draw(win)

	for bird in birds:
		bird.draw(win)
	pygame.display.update()

def fitness(genomes, config):
	global Gen
	Gen += 1

	birds = []
	networks = []
	genome = []

	for _, g in genomes:
		network = neat.nn.FeedForwardNetwork.create(g, config)
		networks.append(network)
		birds.append(Bird(200, 200))
		genome.append(g)
		g.fitness = 0

	ground = Ground(730)
	pipes = [Pipe(700)]
	win = pygame.display.set_mode((Window_width, Window_height))
	clock = pygame.time.Clock()
	running = True
	score = 0

	while running:
		clock.tick(30)
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
				pygame.quit()
				quit()

		pipe_index = 0
		if len(birds) > 0:
			if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].Pipe_Top.get_width():
				pipe_index = 1
		else:
			running = False
			break

		for x, bird in enumerate(birds):
			bird.move()
			genome[x].fitness += 0.1

			output = networks[x].activate((bird.y ,abs(bird.y - pipes[pipe_index].height), abs(bird.y - pipes[pipe_index].bottom)))

			if output[0] > 0.5:
				bird.jump()

		add_pipe = False 
		rem = []
		for pipe in pipes:
			for x, bird in enumerate(birds):
				if pipe.collide(bird):
					genome[x].fitness -= 1
					birds.pop(x)
					genome.pop(x)
					networks.pop(x)

				if not pipe.passed and pipe.x < bird.x:
					pipe.passed = True
					add_pipe = True

			if pipe.x + pipe.Pipe_Top.get_width() < 0:
				rem.append(pipe)
			
			pipe.move()
		
		if add_pipe:
			score +=1

			for g in genome:
				g.fitness +=5

			pipes.append(Pipe(700))

		for r in rem:
			pipes.remove(r)

		for x, bird in enumerate(birds):
			if bird.y + bird.images.get_height() >= 700 or bird.y < 0:
				birds.pop(x)
				genome.pop(x)
				networks.pop(x)

		ground.move()
		draw_window(win, birds, pipes, ground, score, Gen)


def run(config_path):
	config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

	population = neat.Population(config)

	population.add_reporter(neat.StdOutReporter(True))
	population.add_reporter(neat.StatisticsReporter())

	winner = population.run(fitness, 50)

if __name__ == "__main__":
	local_dir = os.path.dirname(__file__)
	config_path = os.path.join(local_dir, "config.txt")
	run(config_path)
