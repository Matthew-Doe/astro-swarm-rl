extends SceneTree

const NeuralPolicyScript = preload("res://simulations/rl/NeuralPolicy.gd")

func _initialize() -> void:
	var arguments := OS.get_cmdline_user_args()
	if arguments.is_empty():
		push_error("Pass a policy JSON path")
		quit(2)
		return
	var policy := NeuralPolicyScript.new()
	var error := policy.load_file(arguments[0])
	if error != OK:
		push_error("Could not load policy: %d" % error)
		quit(3)
		return
	var observation := PackedFloat32Array()
	observation.resize(int(policy.observation["obs_dim"]))
	print("POLICY_RESULT=", JSON.stringify(policy.act(observation)))
	quit()
