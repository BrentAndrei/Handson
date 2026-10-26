import * as fs from 'fs';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { resolveSkeleton } from './src/avatar/boneResolver.js';

console.log('Starting...');
const raw = fs.readFileSync('./public/models/Xbot.glb');
console.log('File read, size:', raw.length);
const buffer = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
const loader = new GLTFLoader();
const gltf = await loader.parseAsync(buffer, '');
console.log('GLTF parsed');
const sk = resolveSkeleton(gltf.scene);
console.log('Skeleton resolved');

const leftForearm = sk.getBone('leftForearm');
const leftHand = sk.getBone('leftHand');
const rightForearm = sk.getBone('rightForearm');
const rightHand = sk.getBone('rightHand');

console.log('=== LEFT ARM ===');
if (leftForearm) {
  console.log('Forearm:', leftForearm.name);
  console.log('  localDir:', leftForearm.localDir.toArray().map(x => x.toFixed(4)));
  console.log('  length:', leftForearm.length.toFixed(4));
  console.log('  children:', leftForearm.children);
} else {
  console.log('leftForearm NOT FOUND');
}
if (leftHand) {
  console.log('Hand:', leftHand.name);
  console.log('  localDir:', leftHand.localDir.toArray().map(x => x.toFixed(4)));
  console.log('  length:', leftHand.length.toFixed(4));
  console.log('  parent:', leftHand.parent);
  console.log('  children:', leftHand.children);
} else {
  console.log('leftHand NOT FOUND');
}

console.log('\n=== RIGHT ARM ===');
if (rightForearm) {
  console.log('Forearm:', rightForearm.name);
  console.log('  localDir:', rightForearm.localDir.toArray().map(x => x.toFixed(4)));
  console.log('  length:', rightForearm.length.toFixed(4));
  console.log('  children:', rightForearm.children);
} else {
  console.log('rightForearm NOT FOUND');
}
if (rightHand) {
  console.log('Hand:', rightHand.name);
  console.log('  localDir:', rightHand.localDir.toArray().map(x => x.toFixed(4)));
  console.log('  length:', rightHand.length.toFixed(4));
  console.log('  parent:', rightHand.parent);
  console.log('  children:', rightHand.children);
} else {
  console.log('rightHand NOT FOUND');
}