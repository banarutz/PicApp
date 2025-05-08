//
//  ColorizerView.swift
//  PicApp
//
//  Created by Sebastian Banaru on 07.01.2025.
//

import SwiftUI
import PhotosUI
import CoreML
import UIKit

struct ColorizerView: View {
    @State private var selectedItem: PhotosPickerItem? = nil
    @State private var originalImage: UIImage? = nil
    @State private var colorizedImage: UIImage? = nil
    @State private var coreModel: MLModel? = nil
    @State private var inferenceTime: Double? = nil
    @State private var showRunButton = false

    var body: some View {
        VStack(spacing: 20) {
            if let originalImage {
                Image(uiImage: originalImage)
                    .resizable()
                    .scaledToFit()
                    .frame(maxHeight: 256)
                    .overlay(Text("Original").foregroundColor(.white).background(Color.black.opacity(0.6)), alignment: .top)
            }

            if let colorizedImage {
                Image(uiImage: colorizedImage)
                    .resizable()
                    .scaledToFit()
                    .frame(maxHeight: 256)
                    .overlay(Text("Colorized").foregroundColor(.white).background(Color.black.opacity(0.6)), alignment: .top)
            }

            if let time = inferenceTime {
                Text(String(format: "Elapsed time: %.2f s", time))
                    .font(.subheadline)
                    .foregroundColor(.gray)
            }

            PhotosPicker("Load Image", selection: $selectedItem, matching: .images)
                .buttonStyle(.borderedProminent)

            if showRunButton {
                Button("Run Colorization") {
                    if let image = originalImage, let model = coreModel {
                        let start = CFAbsoluteTimeGetCurrent()
                        colorizedImage = runModel(on: image, with: model)
                        let end = CFAbsoluteTimeGetCurrent()
                        inferenceTime = (end - start)
                    }
                }
                .disabled(originalImage == nil || coreModel == nil)
                .padding()
                .background(Color.blue)
                .foregroundColor(.white)
                .cornerRadius(10)
            }
        }
        .onAppear {
            Task {
                    await loadModelAsync()
                }
        }
        .onChange(of: selectedItem) {
            Task {
                if let data = try? await selectedItem?.loadTransferable(type: Data.self),
                   let uiImage = UIImage(data: data) {
                    originalImage = uiImage
                    colorizedImage = nil
                    inferenceTime = nil
                    showRunButton = true
                }
            }
        }
    }

    func loadModelAsync() async {
        guard let url = Bundle.main.url(forResource: "colorizer_model", withExtension: "mlmodelc") else {
            print("Model .mlmodelc not found in bundle.")
            return
        }

        do {
            coreModel = try MLModel(contentsOf: url)
            print("Model loaded from bundle.")
        } catch {
            print("Failed to load model: \(error)")
        }
    }

    func runModel(on image: UIImage, with model: MLModel) -> UIImage? {
        guard let resizedImage = image.resize(to: CGSize(width: 256, height: 256)),
              let buffer = resizedImage.toCVPixelBuffer() else {
            return nil
        }

        guard let output = try? model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image_small": buffer])) else {
            print("Model prediction failed.")
            return nil
        }

        guard let multiArray = output.featureValue(for: "colorized_small_image")?.multiArrayValue else {
            print("Missing output array.")
            return nil
        }

        return multiArray.toUIImage()
    }
}

extension UIImage {
    func resize(to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, 0.0)
        self.draw(in: CGRect(origin: .zero, size: size))
        let resized = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resized
    }

    func toCVPixelBuffer() -> CVPixelBuffer? {
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!
        ] as CFDictionary

        var pixelBuffer: CVPixelBuffer?
        let width = Int(self.size.width)
        let height = Int(self.size.height)

        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                         kCVPixelFormatType_32ARGB, attrs,
                                         &pixelBuffer)

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }

        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        let pixelData = CVPixelBufferGetBaseAddress(buffer)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(data: pixelData,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                      space: colorSpace,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
        else {
            CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
            return nil
        }

        UIGraphicsPushContext(context)
        self.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
        UIGraphicsPopContext()
        CVPixelBufferUnlockBaseAddress(buffer, .readOnly)

        return buffer
    }
}


extension MLMultiArray {
    func toUIImage() -> UIImage? {
        guard self.dataType == .float32 || self.dataType == .double else {
            print("Unsupported MLMultiArray data type.")
            return nil
        }

        let shape = self.shape.map { $0.intValue }
        guard shape.count == 4, shape[0] == 1, shape[1] == 3 else {
            print("Unexpected shape \(shape)")
            return nil
        }

        let height = shape[2]
        let width = shape[3]
        let channelStride = height * width

        let floatPtr = UnsafeMutablePointer<Float32>(OpaquePointer(self.dataPointer))

        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: width * height * 4)
        defer { buffer.deallocate() }

        for i in 0..<(width * height) {
            let r = floatPtr[i]
            let g = floatPtr[i + channelStride]
            let b = floatPtr[i + 2 * channelStride]

            buffer[i * 4 + 0] = UInt8(clamping: Int(r))
            buffer[i * 4 + 1] = UInt8(clamping: Int(g))
            buffer[i * 4 + 2] = UInt8(clamping: Int(b))
            buffer[i * 4 + 3] = 255 // Alpha
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerRow = width * 4

        guard let context = CGContext(data: buffer,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: bytesPerRow,
                                      space: colorSpace,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
            print("Failed to create CGContext")
            return nil
        }

        guard let cgImage = context.makeImage() else {
            print("Failed to create CGImage")
            return nil
        }

        return UIImage(cgImage: cgImage, scale: 1.0, orientation: .downMirrored)
    }
}
