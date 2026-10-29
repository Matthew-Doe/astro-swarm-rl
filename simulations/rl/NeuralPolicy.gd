class_name NeuralPolicy
extends RefCounted

var role: String
var profile: String
var simulator: Dictionary = {}
var observation: Dictionary
var network: Dictionary


func load_file(path: String) -> Error:
	var file := FileAccess.open(path, FileAccess.READ)
	if file == null:
		return FileAccess.get_open_error()
	var parsed: Variant = JSON.parse_string(file.get_as_text())
	if not parsed is Dictionary or parsed.get("format", "") != "astroswarm-policy":
		return ERR_FILE_CORRUPT
	role = str(parsed["role"])
	profile = str(parsed.get("profile", "balanced_self_play"))
	simulator = parsed.get("simulator", {})
	observation = parsed["observation"]
	network = parsed["network"]
	return OK


func act(features: PackedFloat32Array) -> Vector2:
	if features.size() != int(observation["obs_dim"]):
		push_error("Policy expected %d observation values, received %d" % [observation["obs_dim"], features.size()])
		return Vector2.ZERO

	var self_dim := int(observation["self_dim"])
	var entity_count := int(observation["entity_count"])
	var entity_dim := int(observation["entity_dim"])
	var mask_index := int(observation["mask_index"])
	var self_encoded := _stack(features.slice(0, self_dim), network["self"])
	var pooled := PackedFloat32Array()
	pooled.resize(64)
	var visible_count := 0.0
	for entity_index in entity_count:
		var start := self_dim + entity_index * entity_dim
		var entity := features.slice(start, start + entity_dim)
		var mask := clampf(entity[mask_index], 0.0, 1.0)
		if mask <= 0.0:
			continue
		var encoded := _stack(entity, network["entity"])
		for index in pooled.size():
			pooled[index] += encoded[index] * mask
		visible_count += mask
	if visible_count > 1.0:
		for index in pooled.size():
			pooled[index] /= visible_count

	var joined := PackedFloat32Array()
	joined.append_array(self_encoded)
	joined.append_array(pooled)
	var hidden := _spiking(joined) if network.has("spiking") else _stack(joined, network["trunk"])
	var action := _dense(hidden, network["output"])
	return Vector2(action[0], action[1])


func _stack(values: PackedFloat32Array, layers: Array) -> PackedFloat32Array:
	var output := values
	for layer: Dictionary in layers:
		output = _dense(output, layer)
	return output


func _dense(values: PackedFloat32Array, layer: Dictionary) -> PackedFloat32Array:
	var input_size := int(layer["input"])
	var output_size := int(layer["output"])
	var weights: Array = layer["weight"]
	var bias: Array = layer["bias"]
	var output := PackedFloat32Array()
	output.resize(output_size)
	for row in output_size:
		var value := float(bias[row])
		var offset := row * input_size
		for column in input_size:
			value += float(weights[offset + column]) * values[column]
		output[row] = tanh(value) if layer.get("activation", "") == "tanh" else value
	return output


func _spiking(values: PackedFloat32Array) -> PackedFloat32Array:
	var settings: Dictionary = network["spiking"]
	var layers: Array = network["trunk"]
	var current := _dense(values, layers[0])
	var membrane1 := PackedFloat32Array()
	var membrane2 := PackedFloat32Array()
	var count := PackedFloat32Array()
	membrane1.resize(current.size())
	membrane2.resize(int(layers[1]["output"]))
	count.resize(membrane2.size())
	var steps := int(settings["steps"])
	var beta := float(settings["beta"])
	var threshold := float(settings["threshold"])
	for tick in steps:
		var spikes := PackedFloat32Array()
		spikes.resize(current.size())
		for index in current.size():
			membrane1[index] = beta * membrane1[index] + current[index]
			if membrane1[index] >= threshold:
				spikes[index] = 1.0
				membrane1[index] -= threshold
		var next_current := _dense(spikes, layers[1])
		for index in membrane2.size():
			membrane2[index] = beta * membrane2[index] + next_current[index]
			if membrane2[index] >= threshold:
				count[index] += 1.0
				membrane2[index] -= threshold
	for index in count.size():
		count[index] /= steps
	return count
