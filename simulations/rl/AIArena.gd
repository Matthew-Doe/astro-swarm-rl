extends Node2D

const NeuralPolicyScript = preload("res://simulations/rl/NeuralPolicy.gd")
const ARENA := Vector2(3840.0, 2160.0)
const PLANET := Vector2(1920.0, 1080.0)
const PLANET_RADIUS := 120.0
const GOAL_RADIUS := 136.0
const SPAWN_RADIUS := 1000.0
const MAX_DEFENDERS := 6
const SELF_DIM := 12
const INTERCEPTOR_ENTITY_DIM := 11
const ATTACKER_ENTITY_DIM := 13
const ACTION_REPEAT := 4

var rng := RandomNumberGenerator.new()
var defenders: Array[Dictionary] = []
var attacker: Dictionary
var interceptor_policy: RefCounted
var attacker_policy: RefCounted
var interceptor_actions: Array[Vector2] = []
var attacker_action := Vector2(1.0, 0.0)
var ticks := 0
var detected := false
var finished := false
var status := "Running"
var quit_after_match := false
var arena_profile := "balanced_self_play"
var defender_count := 5
var defender_fov_min_degrees := 35.0
var defender_fov_max_degrees := 75.0
var paused := false
var finished_at_msec := -1
var match_count := 0
var interceptor_wins := 0
var attacker_goals := 0
var timeouts := 0
var attacker_trail := PackedVector2Array()
var defender_trails: Array[PackedVector2Array] = []


func _ready() -> void:
	var options := _arguments()
	rng.seed = int(options.get("seed", "1"))
	quit_after_match = options.has("headless-match")
	interceptor_policy = _load_policy(str(options.get("interceptor-policy", "")), "interceptor")
	attacker_policy = _load_policy(str(options.get("attacker-policy", "")), "attacker")
	var policy_profile: String = interceptor_policy.profile if interceptor_policy != null else (
		attacker_policy.profile if attacker_policy != null else "balanced_self_play"
	)
	arena_profile = str(options.get("profile", policy_profile))
	var simulator: Dictionary = interceptor_policy.simulator if interceptor_policy != null else (
		attacker_policy.simulator if attacker_policy != null else {}
	)
	defender_fov_min_degrees = float(simulator.get("defender_fov_min_degrees", 35.0))
	defender_fov_max_degrees = float(simulator.get("defender_fov_max_degrees", 75.0))
	defender_count = clampi(int(options.get("defenders", "5")), 1, MAX_DEFENDERS)
	_reset_match()
	set_physics_process(true)


func _arguments() -> Dictionary:
	var result := {}
	for argument in OS.get_cmdline_user_args():
		var clean := argument.trim_prefix("--")
		var split := clean.split("=", true, 1)
		result[split[0]] = split[1] if split.size() == 2 else true
	return result


func _load_policy(path: String, expected_role: String) -> RefCounted:
	if path.is_empty():
		return null
	var policy := NeuralPolicyScript.new()
	var error := policy.load_file(path)
	if error != OK:
		push_error("Could not load policy %s (error %d)" % [path, error])
		return null
	if policy.role != expected_role:
		push_error("Policy %s has role %s, expected %s" % [path, policy.role, expected_role])
		return null
	return policy


func _reset_match() -> void:
	ticks = 0
	detected = false
	finished = false
	finished_at_msec = -1
	status = "Running"
	defenders.clear()
	interceptor_actions.clear()
	attacker_trail = PackedVector2Array()
	defender_trails.clear()
	for index in defender_count:
		var position := PLANET
		for attempt in 40:
			var angle := rng.randf_range(-PI, PI)
			var radius := rng.randf_range(170.0, 250.0)
			position = PLANET + Vector2.from_angle(angle) * radius
			var clear := true
			for prior in defenders:
				if position.distance_to(prior["position"]) < 110.0:
					clear = false
					break
			if clear:
				break
		var sensor_budget := rng.randf()
		defenders.append({
			"position": position,
			"heading": rng.randf_range(-PI, PI),
			"velocity": Vector2.ZERO,
			"speed": 136.0 if arena_profile == "farp_parity" else rng.randf_range(130.0, 180.0),
			"turn_rate": 3.2 if arena_profile == "farp_parity" else rng.randf_range(2.8, 3.8),
			"view": 180.0 if arena_profile == "farp_parity" else 180.0 + 120.0 * sensor_budget,
			"fov": deg_to_rad(50.0 if arena_profile == "farp_parity" else lerpf(defender_fov_max_degrees, defender_fov_min_degrees, sensor_budget)),
			"sees": false,
			"previous": Vector2.ZERO,
		})
		interceptor_actions.append(Vector2(1.0, -0.625))
		defender_trails.append(PackedVector2Array())
	var attacker_angle := rng.randf_range(-PI, PI)
	attacker = {
		"position": PLANET + Vector2.from_angle(attacker_angle) * SPAWN_RADIUS,
		"heading": wrapf(attacker_angle + PI, -PI, PI),
		"velocity": Vector2.ZERO,
		"speed": 150.0 if arena_profile == "farp_parity" else rng.randf_range(135.0, 165.0),
		"turn_rate": 2.8 if arena_profile == "farp_parity" else rng.randf_range(2.5, 3.1),
		"previous": Vector2.ZERO,
	}
	queue_redraw()


func _physics_process(_delta: float) -> void:
	if finished:
		if not quit_after_match and finished_at_msec >= 0 and Time.get_ticks_msec() - finished_at_msec > 3000:
			_reset_match()
			queue_redraw()
		return
	if paused:
		queue_redraw()
		return
	if ticks % ACTION_REPEAT == 0:
		_choose_actions()
	for index in defenders.size():
		_move(defenders[index], interceptor_actions[index], 1.0 / 60.0, 14.0)
	_move(attacker, attacker_action, 1.0 / 60.0, 20.0)
	ticks += 1
	attacker_trail.append(_screen(attacker["position"]))
	if attacker_trail.size() > 180:
		attacker_trail.remove_at(0)
	for index in defenders.size():
		defender_trails[index].append(_screen(defenders[index]["position"]))
		if defender_trails[index].size() > 120:
			defender_trails[index].remove_at(0)
	_update_detection()
	_check_terminal()
	queue_redraw()


func _choose_actions() -> void:
	for index in defenders.size():
		if interceptor_policy != null:
			interceptor_actions[index] = interceptor_policy.act(_interceptor_observation(index))
		elif defenders[index]["sees"]:
			var attacker_position: Vector2 = attacker["position"]
			var defender_position: Vector2 = defenders[index]["position"]
			var relative: float = (attacker_position - defender_position).angle() - float(defenders[index]["heading"])
			interceptor_actions[index] = Vector2(1.0, clampf(sin(relative) * 2.0, -1.0, 1.0))
	if attacker_policy != null:
		attacker_action = attacker_policy.act(_attacker_observation())


func _move(ship: Dictionary, action: Vector2, delta: float, margin: float) -> void:
	action = action.clamp(Vector2(-1.0, -1.0), Vector2.ONE)
	ship["heading"] = wrapf(float(ship["heading"]) + action.y * float(ship["turn_rate"]) * delta, -PI, PI)
	ship["velocity"] = Vector2.from_angle(float(ship["heading"])) * float(ship["speed"]) * action.x
	var position: Vector2 = ship["position"] + ship["velocity"] * delta
	ship["position"] = position.clamp(Vector2(margin, margin), ARENA - Vector2(margin, margin))
	ship["previous"] = action


func _can_see(defender: Dictionary, target: Vector2) -> bool:
	var offset: Vector2 = target - defender["position"]
	if offset.length_squared() > float(defender["view"]) ** 2:
		return false
	return Vector2.from_angle(float(defender["heading"])).dot(offset.normalized()) >= cos(float(defender["fov"]) * 0.5)


func _update_detection() -> void:
	for defender in defenders:
		defender["sees"] = _can_see(defender, attacker["position"])
		detected = detected or defender["sees"]


func _check_terminal() -> void:
	for defender in defenders:
		if defender["position"].distance_squared_to(attacker["position"]) <= 18.0 ** 2:
			_finish("Interceptors captured the attacker")
			return
	if attacker["position"].distance_squared_to(PLANET) <= GOAL_RADIUS ** 2:
		_finish("Attacker reached the goal%s" % (" undetected" if not detected else ""))
	elif ticks >= 30 * 60:
		_finish("Timeout: interceptors defended the goal")


func _finish(message: String) -> void:
	finished = true
	status = message
	finished_at_msec = Time.get_ticks_msec()
	match_count += 1
	if message.begins_with("Interceptors"):
		interceptor_wins += 1
	elif message.begins_with("Attacker"):
		attacker_goals += 1
	else:
		timeouts += 1
	print("ASTROSWARM_RESULT=", JSON.stringify({"result": message, "ticks": ticks, "detected": detected}))
	if quit_after_match:
		get_tree().quit()


func _unhandled_input(event: InputEvent) -> void:
	if not event is InputEventKey or not event.pressed or event.echo:
		return
	if event.keycode == KEY_SPACE:
		paused = not paused
	elif event.keycode == KEY_R:
		_reset_match()


func _self_observation(ship: Dictionary) -> PackedFloat32Array:
	var position: Vector2 = ship["position"]
	var heading := float(ship["heading"])
	var previous: Vector2 = ship["previous"]
	return PackedFloat32Array([
		(position.x - PLANET.x) / SPAWN_RADIUS, (position.y - PLANET.y) / SPAWN_RADIUS,
		sin(heading), cos(heading), position.x / ARENA.x, (ARENA.x - position.x) / ARENA.x,
		position.y / ARENA.y, (ARENA.y - position.y) / ARENA.y,
		1.0 - float(ticks) / (30.0 * 60.0), float(defenders.size()) / 6.0,
		previous.x, previous.y,
	])


func _interceptor_observation(index: int) -> PackedFloat32Array:
	var observer: Dictionary = defenders[index]
	var result := _self_observation(observer)
	for slot in MAX_DEFENDERS:
		var target: Dictionary
		var enemy := slot == 0
		if enemy:
			target = attacker
		else:
			var others: Array[int] = []
			for candidate in defenders.size():
				if candidate != index:
					others.append(candidate)
			if slot - 1 >= others.size():
				result.append_array(_zeros(INTERCEPTOR_ENTITY_DIM))
				continue
			target = defenders[others[slot - 1]]
		if not _can_see(observer, target["position"]):
			result.append_array(_zeros(INTERCEPTOR_ENTITY_DIM))
			continue
		var relative := _relative_features(observer, target)
		result.append_array(PackedFloat32Array([
			relative[0], relative[1], relative[2], relative[3], relative[4], relative[5],
			relative[6], relative[7], relative[8], 1.0 if enemy else -1.0, 1.0,
		]))
	return result


func _attacker_observation() -> PackedFloat32Array:
	var result := _self_observation(attacker)
	for slot in MAX_DEFENDERS:
		if slot >= defenders.size():
			result.append_array(_zeros(ATTACKER_ENTITY_DIM))
			continue
		var defender: Dictionary = defenders[slot]
		var relative := _relative_features(attacker, defender)
		result.append_array(PackedFloat32Array([
			relative[0], relative[1], relative[2], relative[3], relative[4], relative[5],
			relative[6], relative[7], relative[8], float(defender["view"]) / 500.0,
			float(defender["fov"]) / PI, 1.0 if defender["sees"] else 0.0, 1.0,
		]))
	return result


func _relative_features(observer: Dictionary, target: Dictionary) -> PackedFloat32Array:
	var offset: Vector2 = target["position"] - observer["position"]
	var distance := maxf(offset.length(), 0.000001)
	var local := offset.rotated(-float(observer["heading"])) / distance
	var relative_heading := float(target["heading"]) - float(observer["heading"])
	var velocity: Vector2 = (target["velocity"] - observer["velocity"]) / 400.0
	return PackedFloat32Array([
		offset.x / SPAWN_RADIUS, offset.y / SPAWN_RADIUS, velocity.x, velocity.y,
		sin(relative_heading), cos(relative_heading), distance / SPAWN_RADIUS, local.y, local.x,
	])


func _zeros(size: int) -> PackedFloat32Array:
	var values := PackedFloat32Array()
	values.resize(size)
	return values


func _screen(point: Vector2) -> Vector2:
	return point / 3.0


func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, ARENA / 3.0), Color("09111f"))
	for trail_index in defender_trails.size():
		if defender_trails[trail_index].size() > 1:
			draw_polyline(defender_trails[trail_index], Color(0.33, 0.75, 1.0, 0.26), 1.5)
	if attacker_trail.size() > 1:
		draw_polyline(attacker_trail, Color(1.0, 0.35, 0.45, 0.42), 2.0)
	draw_circle(_screen(PLANET), GOAL_RADIUS / 3.0, Color("263d64"))
	draw_circle(_screen(PLANET), PLANET_RADIUS / 3.0, Color("4c78b5"))
	for defender in defenders:
		var center := _screen(defender["position"])
		var heading := float(defender["heading"])
		var edge_a := center + Vector2.from_angle(heading - float(defender["fov"]) * 0.5) * float(defender["view"]) / 3.0
		var edge_b := center + Vector2.from_angle(heading + float(defender["fov"]) * 0.5) * float(defender["view"]) / 3.0
		draw_colored_polygon(PackedVector2Array([center, edge_a, edge_b]), Color(0.2, 0.7, 1.0, 0.08))
		_draw_ship(center, heading, Color("55bfff"))
	_draw_ship(_screen(attacker["position"]), float(attacker["heading"]), Color("ff6577"))
	draw_rect(Rect2(16, 16, 430, 72), Color(0.02, 0.04, 0.09, 0.88), true)
	draw_string(ThemeDB.fallback_font, Vector2(28, 42), "%s  |  %.1fs  |  detected: %s" % [status, ticks / 60.0, detected], HORIZONTAL_ALIGNMENT_LEFT, -1, 17, Color.WHITE)
	draw_string(ThemeDB.fallback_font, Vector2(28, 68), "SPACE pause  R reset  |  profile: %s" % arena_profile, HORIZONTAL_ALIGNMENT_LEFT, -1, 14, Color("b9c7e6"))
	draw_rect(Rect2(920, 16, 345, 142), Color(0.02, 0.04, 0.09, 0.88), true)
	draw_string(ThemeDB.fallback_font, Vector2(940, 42), "AI ARENA  ·  MATCH %d" % (match_count + (0 if finished else 1)), HORIZONTAL_ALIGNMENT_LEFT, -1, 19, Color("f4f7ff"))
	draw_string(ThemeDB.fallback_font, Vector2(940, 70), "Interceptors: %d    Attacker goals: %d" % [interceptor_wins, attacker_goals], HORIZONTAL_ALIGNMENT_LEFT, -1, 15, Color("b9c7e6"))
	draw_string(ThemeDB.fallback_font, Vector2(940, 94), "Timeouts: %d    ships: %d" % [timeouts, defender_count], HORIZONTAL_ALIGNMENT_LEFT, -1, 15, Color("b9c7e6"))
	draw_string(ThemeDB.fallback_font, Vector2(940, 124), "Interceptor: %s" % ("neural policy" if interceptor_policy != null else "scripted"), HORIZONTAL_ALIGNMENT_LEFT, -1, 14, Color("70d7ff"))
	draw_string(ThemeDB.fallback_font, Vector2(940, 146), "Attacker: %s" % ("neural policy" if attacker_policy != null else "scripted"), HORIZONTAL_ALIGNMENT_LEFT, -1, 14, Color("ff8694"))


func _draw_ship(center: Vector2, heading: float, color: Color) -> void:
	var points := PackedVector2Array([
		center + Vector2.from_angle(heading) * 8.0,
		center + Vector2.from_angle(heading + 2.45) * 5.0,
		center + Vector2.from_angle(heading - 2.45) * 5.0,
	])
	draw_colored_polygon(points, color)
