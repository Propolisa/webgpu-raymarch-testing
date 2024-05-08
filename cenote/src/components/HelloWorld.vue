<template>
  <div>
    <canvas width="1920" height="880" ref="canvas"></canvas>
    <DatGui
      closeText="Close controls"
      openText="Open controls"
      closePosition="bottom"
    >
      <DatNumber v-model="X_RES" :min="0" :max="1920"></DatNumber>
      <DatNumber v-model="Y_RES" :min="0" :max="1080"></DatNumber>
      <DatFolder label="Performance">
        <component
          :is="`DatNumber`"
          v-for="(value, key) in performance_metrics"
          v-bind:key="key"
          v-model="performance_metrics[key].average"
          :label="key"
      /></DatFolder>
      <DatFolder label="Shader">
        <component
          :is="`DatNumber`"
          v-for="key of single_value_uniform_keys"
          v-bind:key="key"
          v-model="fragment_shader_constants[key].value"
          :label="key"
          :min="fragment_shader_constants[key].range?.start"
          :max="fragment_shader_constants[key].range?.end"
      /></DatFolder>
      <DatFolder label="Application">
        <DatSelect v-model="pictureUrl" :items="pictures" label="Picture" />
        <DatBoolean v-model="SINGLE_MODE" label="Only update on change" />
      </DatFolder>
    </DatGui>
  </div>
</template>

<script>
import { onMounted, onUnmounted, ref, reactive } from "vue";
import { makeShaderDataDefinitions, makeStructuredView } from "webgpu-utils";
import shader from "../modules/raymarch_noise.wgsl?raw";
import { DatNumber } from "@cyrilf/vue-dat-gui";
let wgsl_constant_decl_regex =
  /const\s+(?<constantName>\w+)(:\s*(?<constantType>\w+))?\s*=\s*(?<initialValue>[^;]+);(?:\s*\/\/\s*\{(?<humanReadableName>[^:]+):(?<rangeOrValues>[^\}]+)\})?/;
export default {
  data() {
    return {
      performance_metrics: {},
      SINGLE_MODE: true,
      MAX_FRAMES: 1,
      X_RES: 0,
      Y_RES: 0,
      state: {
        position: { x: 0, y: 0, z: -45.0 },
        velocity: { x: 0, y: 0, z: 0 },
        speed: 0.1,
        rotation: { yaw: 0, pitch: 0 },
        angularVelocity: { yaw: 0, pitch: 0 },
        zoomLevel: 2.0,
        isMouseDown: false,
        lastMousePosition: { x: 0, y: 0 },
        keyState: {},
        MOUSE_X: 0.0,
        MOUSE_Y: 0.0,
        mousePressed: false,
      },
      fragment_shader_constants: {},
      canvas: null,
      // TESTING
      background: "#cdeecc",
      titleColor: "#077d43",
      titleFontSize: 75,
      title: "vue-dat-gui",
      showPicture: true,
      boxShadow: {
        offsetX: 27,
        offsetY: 27,
        blurRadius: 75,
        spreadRadius: 2,
        color: "rgba(3, 23, 6, 1)",
      },
      pictureUrl:
        "https://images.unsplash.com/photo-1516214104703-d870798883c5?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=750&q=80",
      pictures: [
        {
          name: "forest",
          value:
            "https://images.unsplash.com/photo-1516214104703-d870798883c5?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=750&q=80",
        },
        {
          name: "mountain",
          value:
            "https://images.unsplash.com/photo-1526080676457-4544bf0ebba9?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=750&q=80",
        },
        {
          name: "beach",
          value:
            "https://images.unsplash.com/photo-1520942702018-0862200e6873?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=750&q=80",
        },
      ],
    };
  },
  watch: {
    fragment_shader_constants: {
      handler() {
        console.log("Updated a shader constant!");
        this.zeroPerfCounters();
        this.MAX_FRAMES++;
      },
      deep: true,
    },
  },
  computed: {
    single_value_uniform_keys() {
      return Object.entries(this.fragment_shader_constants)
        .filter(([k, v]) => k)
        .map(([k, v]) => k);
    },
  },
  async mounted() {
    this.canvas = this.$refs.canvas;
    this.setupEventListeners();
    this.calculateCanvasSize();
    const shaderText = shader;
    let { output: processed_wgsl, extracted_constant_defs } =
      this.processWgsl(shaderText);

    for (let def of extracted_constant_defs) {
      this.fragment_shader_constants[def.name] = {
        value: def.initial_value,
        ...def,
      };
    }
    this.canvas.width = this.X_RES;
    this.canvas.height = this.Y_RES;

    this.canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      this.state.zoomLevel += event.deltaY * -0.0001;
      this.state.zoomLevel = Math.max(0.1, Math.min(10, this.state.zoomLevel));
    });

    document.addEventListener("keydown", (event) => {
      this.SINGLE_MODE = false;
      if (event.key == "s" && event.ctrlKey) {
        this.saveCanvas();
        event.preventDefault();
      }
    });

    const START_TIME = new Date().getTime();

    // Setting up the GPU Pipeline
    // Checking to make sure browser supports WebGPU
    if (!navigator.gpu) {
      throw new Error("WebGPU not supported on this browser.");
    }

    // WebGPU's representation of the available gpu hardware
    const adapter = await navigator.gpu.requestAdapter(); // returns a promise, so use await
    if (!adapter) {
      throw new Error("No appropriate GPUAdapter found.");
    }

    const context = this.canvas.getContext("webgpu");
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    // The main interface through which most interaction with the GPU happens
    const device = await adapter.requestDevice();
    this.device = device;
    context.configure({
      device: device,
      format: canvasFormat,
    });

    const HALF_WIDTH = 1.0;
    const vertices = new Float32Array([
      -HALF_WIDTH,
      -HALF_WIDTH,
      HALF_WIDTH,
      -HALF_WIDTH,
      HALF_WIDTH,
      HALF_WIDTH,

      -HALF_WIDTH,
      -HALF_WIDTH,
      HALF_WIDTH,
      HALF_WIDTH,
      -HALF_WIDTH,
      HALF_WIDTH,
    ]);

    const vertexBuffer = device.createBuffer({
      label: "Cell vertices",
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    const vertexBufferLayout = {
      arrayStride: 8,
      attributes: [
        {
          format: "float32x2",
          offset: 0,
          shaderLocation: 0,
        },
      ],
    };

    const UNIFORM_BUFFER_SOURCES = [
      [
        "resolution",
        { getter: (e) => [this.X_RES, this.Y_RES], primitive_type: "vec2f" },
      ],
      [
        "time",
        {
          getter: (e) => [(new Date().getTime() - START_TIME) / 1000],
          primitive_type: "f32",
        },
      ],
      [
        "mouse",
        {
          getter: (e) => [this.state.MOUSE_X, this.state.MOUSE_Y],
          primitive_type: "vec2f",
        },
      ],
      [
        "zoom",
        { getter: (e) => [this.state.zoomLevel], primitive_type: "f32" },
      ],
      [
        "upos",
        {
          getter: (e) => [
            this.state.position.x,
            this.state.position.y,
            this.state.position.z,
          ],
          primitive_type: "vec3f",
        },
      ],
      [
        "urot",
        {
          getter: (e) => [this.state.rotation.pitch, this.state.rotation.yaw],
          primitive_type: "vec2f",
        },
      ],
      ...extracted_constant_defs.map((def) => [
        def.name,
        {
          getter: (e) => [this.fragment_shader_constants[def.name].value],
          carrier: def.primitive_type == "f32" ? "Float32Array" : "Int32Array",
          primitive_type: def.primitive_type,
        },
      ]),
    ];
    UNIFORM_BUFFER_SOURCES.forEach(
      (e) => (e[1].primitive_type = e[1].primitive_type || "f32")
    );
    UNIFORM_BUFFER_SOURCES.forEach(
      (e) => (e[1].carrier = e[1].carrier || "Float32Array")
    );
    const GROUPED_BUFFER_SOURCES = Object.groupBy(
      UNIFORM_BUFFER_SOURCES,
      ({ carrier }) => carrier || "Float32Array"
    );

    const typemap = {
      Float32Array: (size) => `array<f32, ${size}>`,
      Int32Array: (size) => `array<i32, ${size}>`,
    };
    const GROUPED_UNIFORM_BUFFERS = Object.entries(GROUPED_BUFFER_SOURCES).map(
      ([buffer_type, strategies]) => {
        function calc_values() {
          let items = strategies.map(
            ([constant_name, { getter, primitive_type }]) => [
              constant_name,
              getter(),
              primitive_type,
            ]
          );
          let composite_array = new window[buffer_type](
            [].concat(...items.map((e) => e[1]))
          );
          return { items, composite_array };
        }
        let { items, composite_array } = calc_values();

        return [
          buffer_type,

          {
            atlas: {
              ...computeIndexRanges(items, {
                buffer_type,
              }),
            },
            num_entries: items.length,
            buffer_type,
            primitive_type: strategies[0][1].primitive_type,
            // BUFFER,
            // update() {
            //   device?.queue?.writeBuffer(
            //     BUFFER,
            //     0,
            //     calc_values()?.composite_array
            //   );
            // },
          },
        ];
      }
    );

    const UNIFORM_BUFFERS = Object.values(GROUPED_UNIFORM_BUFFERS).flatMap(
      (e) => e[1]
    );

    let remapped_constants = [];
    // for (let def of extracted_constant_defs) {
    //   let { length, start, end, buffer_type } = UNIFORM_BUFFERS.map(
    //     (e) => e.atlas[def.name]
    //   ).find(Boolean);
    //   remapped_constants.push(`const ${def.name} = CELO.${def.name};`);
    // }

    let struct_declarations = `struct TransferableConfig {
      ${UNIFORM_BUFFERS.map((e) =>
        Object.entries(e.atlas).map(([k, v]) => `${k}: ${v.primitive_type}`)
      )
        .map((e) => e + ",")
        .join("\n")}
    }`;
    let uniform_declarations = `@group(0) @binding(0) var<uniform> C: TransferableConfig;`;

    console.log("SHADER UNIFORMS:", "\n" + uniform_declarations);

    const GPU_UNIFORMS = Object.fromEntries(UNIFORM_BUFFERS);

    const PROCESSED_FRAGMENT_SHADER_CODE = [
      struct_declarations,
      uniform_declarations,
      remapped_constants.join("\n"),
      processed_wgsl,
    ].join("\n\n");

    let defs = makeShaderDataDefinitions(
      [
        struct_declarations,
        uniform_declarations,
        remapped_constants.join("\n"),
      ].join("\n\n")
    );

    const UNIFORMS = makeStructuredView(defs.uniforms["C"]);

    const uniformBuffer = device.createBuffer({
      size: UNIFORMS.arrayBuffer.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    let UPDATE_UNIFORMS = () => {
      UNIFORMS.set(
        Object.fromEntries(
          GROUPED_BUFFER_SOURCES.Float32Array.map(([k, v]) => [k, v.getter()])
        )
      );
      device.queue.writeBuffer(uniformBuffer, 0, UNIFORMS.arrayBuffer);
    };

    const cellShaderModule = device.createShaderModule({
      label: "Cell shader",
      code: PROCESSED_FRAGMENT_SHADER_CODE,
    });

    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
      ],
    });

    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    });

    const cellPipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: cellShaderModule,
        entryPoint: "vertexMain",
        buffers: [vertexBufferLayout],
      },
      fragment: {
        module: cellShaderModule,
        entryPoint: "fragmentMain",
        targets: [{ format: canvasFormat }],
      },
    });

    const bindGroup = device.createBindGroup({
      layout: cellPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
    });

    let frameCount = 0;
    let lastTime = performance.now();

    const draw = () => {
      const run = async () => {
        const perf = this.performanceEvaluator;
        this.canvas.width = this.X_RES;
        this.canvas.height = this.Y_RES;
        // Inline use of performanceEvaluator for each function block
        await perf("update_movement", this.updateMovement);

        device.queue.writeBuffer(vertexBuffer, 0, vertices);

        UPDATE_UNIFORMS();

        const encoder = device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
          colorAttachments: [
            {
              view: context.getCurrentTexture().createView(),
              loadOp: "clear",
              clearValue: { r: 0, g: 0, b: 0.4, a: 1 },
              storeOp: "store",
            },
          ],
        });

        pass.setPipeline(cellPipeline);
        pass.setVertexBuffer(0, vertexBuffer);
        pass.setBindGroup(0, bindGroup);

        pass.draw(vertices.length / 2);

        pass.end();
        device.queue.submit([encoder.finish()]);
        perf("render", (e) => device.queue.onSubmittedWorkDone());
      };
      if (this.SINGLE_MODE && frameCount >= this.MAX_FRAMES) {
      } else {
        run();
        frameCount++;
      }
      requestAnimationFrame(draw);
    };
    draw();
  },
  unmounted() {
    this.removeEventListeners();
    this.device?.destroy();
  },
  methods: {
    zeroPerfCounters() {
      Object.keys(this.performance_metrics).forEach((k) => {
        this.performance_metrics[k].count = 0;
        this.performance_metrics[k].total_time = 0;
      });
    },
    async performanceEvaluator(key, fn) {
      // Ensure the metric key is registered
      if (!this.performance_metrics[key]) {
        this.performance_metrics[key] = { total_time: 0, count: 0 };
      }

      const start = performance.now();
      const result = await fn();
      const end = performance.now();

      let metric_data = this.performance_metrics[key];
      metric_data.total_time += end - start;
      metric_data.count += 1;
      metric_data.average = metric_data.total_time / metric_data.count;

      return result;
    },
    extract_const_decl(line) {
      let matches = line.match(wgsl_constant_decl_regex);
      if (matches) {
        let extracted_data = {
          name: matches.groups.constantName,
          primitive_type: matches.groups.constantType || null, // If type is not provided, default to 'Implicit'
          initial_value: matches.groups.initialValue.trim(),
          human_name: matches.groups.humanReadableName
            ? matches.groups.humanReadableName.trim()
            : null,
          __range_or_values: matches.groups.rangeOrValues
            ? matches.groups.rangeOrValues.trim()
            : null,
        };

        let { __range_or_values: rov } = extracted_data;
        if (rov) {
          if (rov.startsWith("[")) {
            extracted_data.value_options = rov
              .slice(1, -1)
              .split(",")
              .map((e) => e.trim())
              .map(Number);
          } else if (rov.includes("-")) {
            let [start, end] = rov
              .split("-")
              .map((e) => e.trim())
              .slice(0, 2)
              .map(Number);
            extracted_data.range = { start, end };
          }
        }
        if (!extracted_data.primitive_type) return null;
        // extracted_data.primitive_type = extracted_data.initial_value.includes(
        //   "."
        // )
        //   ? "f32"
        //   : "i32";
        extracted_data.initial_value = Number(
          extracted_data.initial_value.trim()
        );
        extracted_data.validator = !extracted_data.primitive_type.startsWith(
          "f"
        )
          ? (val) => Math.floor(val)
          : (val) => val;
        return extracted_data;
      } else {
        return null; // Return null if the line does not match the pattern
      }
    },

    processWgsl(wgslText) {
      let buffer = [];
      let extracted_constant_defs = [];
      wgslText.split("\n").forEach((line) => {
        if (line.startsWith("const ")) {
          // it's a module top level constant
          let constant_def = this.extract_const_decl(line);
          if (constant_def?.primitive_type) {
            extracted_constant_defs.push(constant_def);
          } else {
            console.warn(
              `Trouble parsing constant declaration '${line}' -- disregarding`
            );
            buffer.push(line);
          }
        } else if (line.startsWith("@group(0) @binding(")) {
          // drop, we are gonna dynamically push the bindings back in later
        } else {
          buffer.push(line);
        }
      });

      return { output: buffer.join("\n"), extracted_constant_defs };
    },
    calculateCanvasSize() {
      const padding = 32;
      const windowWidth = window.innerWidth;
      const windowHeight = window.innerHeight;
      const aspectRatio = windowWidth / windowHeight;

      let canvasWidth = windowWidth - 2 * padding;
      let canvasHeight = canvasWidth / aspectRatio;

      if (canvasHeight > windowHeight - 2 * padding) {
        canvasHeight = windowHeight - 2 * padding;
        canvasWidth = canvasHeight * aspectRatio;
      }

      this.X_RES = Math.floor(canvasWidth);
      this.Y_RES = Math.floor(canvasHeight);

      this.$refs.canvas.width = this.X_RES;
      this.$refs.canvas.height = this.Y_RES;
    },
    updateMovement() {
      // Reset movement accumulators
      let moveX = 0,
        moveY = 0,
        moveZ = 0,
        rotateYaw = 0,
        rotatePitch = 0;

      // Get trigonometric values of yaw and pitch for movement direction calculations
      const cosYaw = Math.cos(this.state.rotation.yaw);
      const sinYaw = Math.sin(this.state.rotation.yaw);
      const cosPitch = Math.cos(this.state.rotation.pitch);
      const sinPitch = Math.sin(this.state.rotation.pitch);

      // Read keyboard input
      const { keyState, speed } = this.state;
      if (keyState["w"]) {
        moveX += speed * cosPitch * sinYaw;
        moveY += speed * sinPitch;
        moveZ += speed * cosPitch * cosYaw;
      }
      if (keyState["s"]) {
        moveX -= speed * cosPitch * sinYaw;
        moveY -= speed * sinPitch;
        moveZ -= speed * cosPitch * cosYaw;
      }
      if (keyState["a"]) {
        moveX -= speed * cosYaw;
        moveZ += speed * sinYaw;
      }
      if (keyState["d"]) {
        moveX += speed * cosYaw;
        moveZ -= speed * sinYaw;
      }

      // Read gamepad input
      const gamepads = navigator.getGamepads ? navigator.getGamepads() : [];
      for (let gamepad of gamepads) {
        if (gamepad) {
          // Adjust movement based on the gamepad left stick and yaw + pitch rotation
          moveX +=
            gamepad.axes[0] * speed * cosYaw -
            gamepad.axes[1] * speed * sinYaw * cosPitch;
          moveY += gamepad.axes[1] * speed * sinPitch;
          moveZ -=
            gamepad.axes[1] * speed * cosYaw * cosPitch +
            gamepad.axes[0] * speed * sinYaw;
          rotateYaw += gamepad.axes[2] * 0.01; // Assuming gamepad axes[2] controls yaw rotation
          rotatePitch += gamepad.axes[3] * 0.01; // Assuming gamepad axes[3] controls pitch rotation
        }
      }

      // Update velocities
      this.state.velocity.x += moveX;
      this.state.velocity.y += moveY;
      this.state.velocity.z += moveZ;
      this.state.angularVelocity.yaw += rotateYaw;
      this.state.angularVelocity.pitch += rotatePitch;

      // Apply dampening
      this.state.velocity.x *= 0.95;
      this.state.velocity.y *= 0.95;
      this.state.velocity.z *= 0.95;
      this.state.angularVelocity.yaw *= 0.9;
      this.state.angularVelocity.pitch *= 0.9;

      // Apply velocity to position
      this.state.position.x += this.state.velocity.x;
      this.state.position.y += this.state.velocity.y;
      this.state.position.z += this.state.velocity.z;

      // Update rotation and clamp the pitch to avoid gimbal lock or extreme flips
      this.state.rotation.yaw += this.state.angularVelocity.yaw;
      this.state.rotation.pitch += this.state.angularVelocity.pitch;
      // const MAX_PITCH = Math.PI / 2 - 0.1; // slightly less than 90 degrees
      // this.state.rotation.pitch = Math.max(
      //   -MAX_PITCH,
      //   Math.min(MAX_PITCH, this.state.rotation.pitch)
      // );
    },
    handleKeyDown(event) {
      this.state.keyState[event.key.toLowerCase()] = true;
    },
    handleKeyUp(event) {
      this.state.keyState[event.key.toLowerCase()] = false;
    },
    handleMouseDown(event) {
      this.state.isMouseDown = true;
      this.state.lastMousePosition = { x: event.clientX, y: event.clientY };
    },
    handleMouseMove(event) {
      if (!this.state.isMouseDown) return;

      const deltaX = event.clientX - this.state.lastMousePosition.x;
      const deltaY = event.clientY - this.state.lastMousePosition.y;

      // Instead of directly changing yaw and pitch, adjust angular velocity
      this.state.angularVelocity.yaw += deltaX * 0.0005;
      this.state.angularVelocity.pitch += deltaY * 0.0005;

      this.state.lastMousePosition = { x: event.clientX, y: event.clientY };
    },
    handleMouseUp() {
      this.state.isMouseDown = false;
    },
    updateMouse(e) {
      const domain = this.canvas.getBoundingClientRect();
      if (this.state.mousePressed) {
        this.state.MOUSE_X = e.clientX - domain.left - this.X_RES / 2;
        this.state.MOUSE_Y = e.clientY - domain.top - this.Y_RES / 2;
      }
    },
    saveCanvas(e) {
      const link = document.createElement("a");
      link.download = "shader.png";
      link.href = this.canvas.toDataURL();
      link.click();
      link.delete;
    },
    handleGamepadConnected(event) {
      console.log(
        `Gamepad connected at index ${event.gamepad.index}: ${event.gamepad.id}. ${event.gamepad.buttons.length} buttons, ${event.gamepad.axes.length} axes.`
      );
      this.updateGamepadState(); // Start reading gamepad state
    },
    handleGamepadDisconnected(event) {
      console.log(
        `Gamepad disconnected from index ${event.gamepad.index}: ${event.gamepad.id}`
      );
    },
    updateGamepadState() {
      const gamepads = navigator.getGamepads ? navigator.getGamepads() : [];
      const threshold = 0.1; // Adjust this threshold value as needed

      for (let gamepad of gamepads) {
        if (gamepad) {
          window.gamepad = gamepad;
          if (Math.abs(gamepad.axes[0]) > threshold) {
            this.state.position.x += gamepad.axes[0] * this.state.speed * 0.01;
          }
          if (Math.abs(gamepad.axes[1]) > threshold) {
            this.state.position.z -= gamepad.axes[1] * this.state.speed * 0.01;
          }
          if (Math.abs(gamepad.axes[2]) > threshold) {
            this.state.rotation.yaw += gamepad.axes[2] * 0.0001;
          }
          if (Math.abs(gamepad.axes[3]) > threshold) {
            this.state.rotation.pitch += gamepad.axes[3] * 0.0001;
          }
        }
      }
      requestAnimationFrame(this.updateGamepadState);
    },
    setupEventListeners() {
      window.addEventListener("keydown", this.handleKeyDown);
      window.addEventListener("keyup", this.handleKeyUp);
      window.addEventListener("gamepadconnected", this.handleGamepadConnected);
      window.removeEventListener(
        "gamepaddisconnected",
        this.handleGamepadDisconnected
      );

      this.canvas.addEventListener("mousedown", this.handleMouseDown);
      this.canvas.addEventListener("mouseup", this.handleMouseUp);
      this.canvas.addEventListener("mouseleave", this.handleMouseUp);
      this.canvas.addEventListener("mousemove", this.handleMouseMove);
      this.canvas.addEventListener("mousemove", this.updateMouse);
    },
    removeEventListeners() {
      window.removeEventListener("keydown", this.handleKeyDown);
      window.removeEventListener("keyup", this.handleKeyUp);
      window.removeEventListener(
        "gamepadconnected",
        this.handleGamepadConnected
      );
      window.removeEventListener(
        "gamepaddisconnected",
        this.handleGamepadDisconnected
      );

      this.canvas.removeEventListener("mousedown", this.handleMouseDown);
      this.canvas.removeEventListener("mouseup", this.handleMouseUp);
      this.canvas.removeEventListener("mouseleave", this.handleMouseUp);
      this.canvas.removeEventListener("mousemove", this.handleMouseMove);
      this.canvas.removeEventListener("mousemove", this.updateMouse);
    },
  },
};

function computeIndexRanges(dataEntries, optionalProps = {}) {
  const indexMap = dataEntries.reduce(
    (acc, [key, array, primitive_type], index) => {
      const start = acc.prevEnd + 1; // Start from the next index after the previous array
      const end = start + array.length - 1; // Calculate the end index
      acc[key] = {
        start,
        end,
        length: array.length,
        ...optionalProps,
        primitive_type,
      }; // Assign the range to the key
      acc.prevEnd = end; // Update the previous end index for the next iteration
      return acc;
    },
    { prevEnd: -1 }
  );

  delete indexMap.prevEnd;
  return indexMap;
}
</script>

<style scoped>
#overlay {
  position: fixed; /* Sit on top of the page content */
  display: none; /* Hidden by default */
  width: 100%; /* Full width (cover the whole page) */
  height: 100%; /* Full height (cover the whole page) */
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.5); /* Black background with opacity */
  z-index: 2; /* Specify a stack order in case you're using a different order for other elements */
  cursor: pointer; /* Add a pointer on hover */
}
canvas {
  border-radius: 15px;
  border: 1px solid #333;
}
</style>
