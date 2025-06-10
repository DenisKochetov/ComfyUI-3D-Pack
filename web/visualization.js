import { app } from "/scripts/app.js"

class Visualizer {
    constructor(node, container, visualSrc) {
        this.node = node

        this.iframe = document.createElement('iframe')
        Object.assign(this.iframe, {
            scrolling: "no",
            overflow: "hidden",
        })
        this.iframe.src = "/extensions/ComfyUI-3D-Pack/html/" + visualSrc + ".html"
        container.appendChild(this.iframe)
    }

    updateVisual(params) {
        console.log("[Comfy3D][Visualizer.updateVisual] Received params:", JSON.parse(JSON.stringify(params || null)));
        const iframeDocument = this.iframe.contentWindow.document
        const previewScript = iframeDocument.getElementById('visualizer')
        
        if (this.iframe.contentWindow && typeof this.iframe.contentWindow.updateSequenceData === 'function') {
            this.iframe.contentWindow.updateSequenceData(params);
        } else {
            console.warn("iframe.contentWindow.updateSequenceData function not found. Sending params as attributes.");
            if (params.filepath) {
                 previewScript.setAttribute("filepath", params.filepath);
            }
            if (params.filepaths) {
                previewScript.setAttribute("filepaths_json", JSON.stringify(params.filepaths));
            }
        }

        const timestamp = Date.now().toString()
        previewScript.setAttribute("timestamp", timestamp)
    }

    remove() {
        this.container.remove()
    }
}

function createVisualizer(node, inputName, typeName, inputData, app) {
    node.name = inputName

    const widget = {
        type: typeName,
        name: "preview3d",
        callback: () => {},
        draw : function(ctx, node, widgetWidth, widgetY, widgetHeight) {
            const margin = 30
            const top_offset = LiteGraph.NODE_TITLE_HEIGHT+margin
            const visible = app.canvas.ds.scale > 0.5 && this.type === typeName

            const [x, y] = node.getBounding();
            const [left, top] = app.canvasPosToClientPos([x, y]);
            const width = node.width * app.canvas.ds.scale;
            const height = (node.height - top_offset ) * app.canvas.ds.scale;

            Object.assign(this.visualizer.style, {
                left: `${left}px`,
                top: `${top+(top_offset * app.canvas.ds.scale)}px`,
                width: `${width}px`,
                height: `${height}px`,
                position: "absolute",
                overflow: "hidden",
            })

            Object.assign(this.visualizer.children[0].style, {
                transformOrigin: "50% 50%",
                width: '100%',
                height: '100%',
                border: '0 none',
            })

            this.visualizer.hidden = !visible
        },
    }

    const container = document.createElement('div')
    container.id = `Comfy3D_${inputName}`

    node.visualizer = new Visualizer(node, container, typeName)
    widget.visualizer = container
    widget.parent = node

    document.body.appendChild(widget.visualizer)

    node.addCustomWidget(widget)

    node.updateParameters = (params) => {
        console.log("[Comfy3D][node.updateParameters] Received params:", JSON.parse(JSON.stringify(params || null)));
        node.visualizer.updateVisual(params)
    }

    // Events for drawing backgound
    node.onDrawBackground = function (ctx) {
        if (!this.flags.collapsed) {
            node.visualizer.iframe.hidden = false
        } else {
            node.visualizer.iframe.hidden = true
        }
    }

    // Make sure visualization iframe is always inside the node when resize the node
    node.onResize = function () {
        let [w, h] = this.size
        if (w <= 600) w = 600
        if (h <= 500) h = 500

        if (w > 600) {
            h = w - 100
        }

        this.size = [w, h]
    }

    // Events for remove nodes
    node.onRemoved = () => {
        for (let w in node.widgets) {
            if (node.widgets[w].visualizer) {
                node.widgets[w].visualizer.remove()
            }
        }
    }


    return {
        widget: widget,
    }
}

function registerVisualizer(nodeType, nodeData, nodeClassName, typeName){
    console.log(`[Comfy3D] Attempting to register: ${nodeData.name}, checking against ${nodeClassName}`);
    if (nodeData.name == nodeClassName) {
        console.log(`[Comfy3D] [SUCCESS] Registering node: ${nodeData.name} with typeName: ${typeName}`);

        const onNodeCreated = nodeType.prototype.onNodeCreated

        nodeType.prototype.onNodeCreated = async function() {
            console.log(`[Comfy3D] onNodeCreated called for: ${this.type}`);
            const r = onNodeCreated
                ? onNodeCreated.apply(this, arguments)
                : undefined

            let nodeName = `Preview3DNode_${this.type.replace(/\s|\[|\]/g, '')}`;

            console.log(`[Comfy3D] Creating visualizer for node: ${nodeName}, type: ${this.type}`);

            const result = await createVisualizer.apply(this, [this, nodeName, typeName, {}, app])

            this.setSize([600, 500])

            return r
        }

        nodeType.prototype.onExecuted = async function(message) {
            console.log("[Comfy3D][nodeType.onExecuted] Received message:", JSON.parse(JSON.stringify(message || null)));
            if (message?.previews) {
                console.log("[Comfy3D][nodeType.onExecuted] Data to send to updateParameters:", JSON.parse(JSON.stringify(message.previews[0] || null)));
                this.updateParameters(message.previews[0])
            }
        }
    }
}

app.registerExtension({
    name: "Mr.ForExample.Visualizer.GS",

    async init (app) {

    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        console.log(`[Comfy3D] beforeRegisterNodeDef: Checking node type: ${nodeData.name} (Display Name might be different)`);
        registerVisualizer(nodeType, nodeData, "[Comfy3D] Preview 3DGS", "gsVisualizer")
        registerVisualizer(nodeType, nodeData, "[Comfy3D] Preview 3DMesh", "threeVisualizer")
        registerVisualizer(nodeType, nodeData, "GLBSequencePreviewNode", "glbSequenceVisualizer")
    },
})