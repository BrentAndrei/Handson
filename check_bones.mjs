import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const loader = new GLTFLoader();
loader.load(join(__dirname, 'public/models/Xbot.glb'), (gltf) => {
  const model = gltf.scene;
  model.updateMatrixWorld(true);
  
  // Find hand and forearm bones
  let leftHand, leftForearm, rightHand, rightForearm;
  model.traverse((child) => {
    if (child.name === 'mixamorig:LeftHand') leftHand = child;
    if (child.name === 'mixamorig:LeftForeArm') leftForearm = child;
    if (child.name === 'mixamorig:RightHand') rightHand = child;
    if (child.name === 'mixamorig:RightForeArm') rightForearm = child;
  });
  
  console.log('=== LEFT ARM ===');
  if (leftForearm) {
    const wPos = new THREE.Vector3();
    leftForearm.getWorldPosition(wPos);
    console.log('LeftForearm world pos:', wPos);
    console.log('LeftForearm children:', leftForearm.children.map(c => c.name));
  }
  if (leftHand) {
    const wPos = new THREE.Vector3();
    leftHand.getWorldPosition(wPos);
    console.log('LeftHand world pos:', wPos);
    console.log('LeftHand parent:', leftHand.parent?.name);
    console.log('LeftHand children:', leftHand.children.map(c => c.name));
  }
  
  console.log('\n=== RIGHT ARM ===');
  if (rightForearm) {
    const wPos = new THREE.Vector3();
    rightForearm.getWorldPosition(wPos);
    console.log('RightForearm world pos:', wPos);
    console.log('RightForearm children:', rightForearm.children.map(c => c.name));
  }
  if (rightHand) {
    const wPos = new THREE.Vector3();
    rightHand.getWorldPosition(wPos);
    console.log('RightHand world pos:', wPos);
    console.log('RightHand parent:', rightHand.parent?.name);
    console.log('RightHand children:', rightHand.children.map(c => c.name));
  }
  
  process.exit(0);
});