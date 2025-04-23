import SwiftUI
import CoreML
import UIKit
import PhotosUI
import CoreImage
import CoreImage.CIFilterBuiltins


struct DenoisingViewController: View {
    @State private var inputImage: UIImage?
    @State private var denoisedImage: UIImage?
    @State private var isProcessing = false
    @State private var pickerDelegate: PickerDelegate?

    var body: some View {
        VStack(spacing: 20) {
            if let denoised = denoisedImage {
                Image(uiImage: denoised)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(height: 300)
                    .overlay(Text("Denoised Image").foregroundColor(.white).background(Color.black.opacity(0.6)), alignment: .top)
            }

            if let input = inputImage {
                Image(uiImage: input)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(height: 300)
                    .overlay(Text("Original Image").foregroundColor(.white).background(Color.black.opacity(0.6)), alignment: .top)

                if denoisedImage == nil {
                    Button("Run Denoising", action: runDenoising)
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }
            } else {
                Button("Select Image") {
                    showImagePicker()
                }
                .padding()
            }
        }
        .padding()
    }

    func showImagePicker() {
        let config = PHPickerConfiguration()
        let picker = PHPickerViewController(configuration: config)
        let delegate = PickerDelegate { image in
            inputImage = image
            denoisedImage = nil
        }
        picker.delegate = delegate
        pickerDelegate = delegate

        if let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
           let root = scene.windows.first?.rootViewController {
            root.present(picker, animated: true)
        }
    }

    func runDenoising() {
        guard let input = inputImage, let cgImage = input.cgImage else { return }
        isProcessing = true

        let width = cgImage.width
        let height = cgImage.height
        let patchSize = 50
        var patches: [(x: Int, y: Int, patch: MLMultiArray)] = []

        for y in stride(from: 0, to: height, by: patchSize) {
            for x in stride(from: 0, to: width, by: patchSize) {
                let patch = extractPatch(from: input, at: x, y: y, size: patchSize)
                if let mlArray = patch {
                    patches.append((x, y, mlArray))
                }
            }
        }

        guard let modelURL = Bundle.main.url(forResource: "dncnn-epoch=028-val_loss=0.0001", withExtension: "mlmodelc"),
              let coremlModel = try? MLModel(contentsOf: modelURL) else {
            print("Model not found")
            return
        }

        let outImage = reconstructImage(from: patches, model: coremlModel, originalSize: CGSize(width: width, height: height), patchSize: patchSize)

        DispatchQueue.main.async {
            denoisedImage = outImage
            isProcessing = false
        }
    }

    func extractPatch(from image: UIImage, at x: Int, y: Int, size: Int) -> MLMultiArray? {
        guard let cgImage = image.cgImage else { return nil }
        let width = cgImage.width
        let height = cgImage.height

        let cropRect = CGRect(x: x, y: y,
                              width: min(size, width - x),
                              height: min(size, height - y))
        guard let croppedCG = cgImage.cropping(to: cropRect) else { return nil }

        UIGraphicsBeginImageContextWithOptions(CGSize(width: size, height: size),
                                               false, 1.0)
        let context = UIGraphicsGetCurrentContext()!
        context.setFillColor(UIColor.black.cgColor)
        context.fill(CGRect(x: 0, y: 0, width: size, height: size))
        UIImage(cgImage: croppedCG)
          .draw(in: CGRect(origin: .zero,
                           size: CGSize(width: cropRect.width,
                                        height: cropRect.height)))
        let paddedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()

        // 1) conversie la MLMultiArray
        guard var patchArray = paddedImage?.toMLMultiArray() else { return nil }
        // 2) normalizeChannels (echivalent ToTensorV2 + Normalize(0,1))
        patchArray.normalizeChannels(mean: [0,0,0], std: [1,1,1])
        return patchArray
    }


    func reconstructImage(
        from patches: [(x: Int, y: Int, patch: MLMultiArray)],
        model: MLModel,
        originalSize: CGSize,
        patchSize: Int
    ) -> UIImage? {
        let width = Int(originalSize.width)
        let height = Int(originalSize.height)
        let colorSpace = CGColorSpaceCreateDeviceRGB()

        guard let context = CGContext(
            data: nil,
            width: width, height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else {
            return nil
        }

        for (x, y, inputArray) in patches {
            // predict: model.forward already returns denoised = x - noise
            guard let outputArray = try? model
                .prediction(from: MLDictionaryFeatureProvider(
                    dictionary: ["input": inputArray]))
                .featureValue(for: "var_252")?
                .multiArrayValue else {
                continue
            }

            // clamp și pregătește array-ul pentru desen
            for i in 0..<outputArray.count {
                let original = inputArray[i].floatValue
                let denoised  = outputArray[i].floatValue
                // 1) clamp în [0,1]
                let clamped = max(0, min(1, denoised))
                // 2) suprascrie cu valoare denormalizată în [0,255]
                inputArray[i] = NSNumber(value: clamped)

                // debug print înainte de scalare
                if i < 5 {
                    print("orig:", original,
                          "denoised:", denoised,
                          "diff:", original - denoised)
                }
            }

            // acum desenăm patch-ul
            if let buffer = inputArray.toCVPixelBuffer(
                    width: patchSize, height: patchSize),
               let img    = UIImage(pixelBuffer: buffer),
               let patchCG = img.cgImage {
                context.draw(patchCG,
                             in: CGRect(x: x, y: y,
                                        width: patchSize, height: patchSize))
            }
        }

        guard let finalCG = context.makeImage() else { return nil }
        return UIImage(cgImage: finalCG,
                       scale: 1.0,
                       orientation: .downMirrored)
    }

}

class PickerDelegate: NSObject, PHPickerViewControllerDelegate {
    let completion: (UIImage?) -> Void
    init(completion: @escaping (UIImage?) -> Void) {
        self.completion = completion
    }
    func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
        picker.dismiss(animated: true)
        guard let provider = results.first?.itemProvider, provider.canLoadObject(ofClass: UIImage.self) else {
            completion(nil)
            return
        }
        provider.loadObject(ofClass: UIImage.self) { object, _ in
            DispatchQueue.main.async {
                self.completion(object as? UIImage)
            }
        }
    }
}

extension UIImage {
    func toCVPixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ] as CFDictionary

        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                         kCVPixelFormatType_32ARGB, attrs,
                                         &pixelBuffer)
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else {
            return nil
        }

        UIGraphicsPushContext(context)
        self.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
        UIGraphicsPopContext()

        return buffer
    }
}

extension MLMultiArray {
    func toCVPixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        var pixelBuffer: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ] as CFDictionary

        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                         kCVPixelFormatType_32BGRA, attrs,
                                         &pixelBuffer)
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }

        CVPixelBufferLockBaseAddress(buffer, [])
        guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else {
            CVPixelBufferUnlockBaseAddress(buffer, [])
            return nil
        }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        for y in 0..<height {
            for x in 0..<width {
                let r = UInt8(max(0, min(1, self[[0, 0, y as NSNumber, x as NSNumber]].floatValue)) * 255.0)
                let g = UInt8(max(0, min(1, self[[0, 1, y as NSNumber, x as NSNumber]].floatValue)) * 255.0)
                let b = UInt8(max(0, min(1, self[[0, 2, y as NSNumber, x as NSNumber]].floatValue)) * 255.0)
                
                let pixelOffset = y * bytesPerRow + x * 4
                let pixelPointer = baseAddress.advanced(by: pixelOffset)

                pixelPointer.storeBytes(of: b, as: UInt8.self)
                pixelPointer.advanced(by: 1).storeBytes(of: g, as: UInt8.self)
                pixelPointer.advanced(by: 2).storeBytes(of: r, as: UInt8.self)
                pixelPointer.advanced(by: 3).storeBytes(of: 255, as: UInt8.self) // alpha
            }
        }

        CVPixelBufferUnlockBaseAddress(buffer, [])
        return buffer
    }
    
    func normalizeChannels(mean: [Float], std: [Float]) {
            assert(mean.count == 3 && std.count == 3, "mean and std must have 3 elements each")

            let height = self.shape[2].intValue
            let width = self.shape[3].intValue

            for c in 0..<3 {
                for y in 0..<height {
                    for x in 0..<width {
                        let index: [NSNumber] = [0, NSNumber(value: c), NSNumber(value: y), NSNumber(value: x)]
                        let value = self[index].floatValue
                        let normalized = (value - mean[c]) / std[c]
                        self[index] = NSNumber(value: normalized)
                    }
                }
            }
        }
}


extension UIImage {
    func toMLMultiArray() -> MLMultiArray? {
        guard let pixelBuffer = self.toCVPixelBuffer(width: 50, height: 50) else { return nil }

        let width = 50
        let height = 50
        guard let array = try? MLMultiArray(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)], dataType: .float32) else { return nil }
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        if let base = CVPixelBufferGetBaseAddress(pixelBuffer) {
            let buffer = base.assumingMemoryBound(to: UInt8.self)
            for y in 0..<height {
                for x in 0..<width {
                    let offset = y * CVPixelBufferGetBytesPerRow(pixelBuffer) + x * 4
                    let r = Float(buffer[offset + 1]) / 255.0
                    let g = Float(buffer[offset + 2]) / 255.0
                    let b = Float(buffer[offset + 3]) / 255.0

                    array[[0, 0, y as NSNumber, x as NSNumber]] = NSNumber(value: r)
                    array[[0, 1, y as NSNumber, x as NSNumber]] = NSNumber(value: g)
                    array[[0, 2, y as NSNumber, x as NSNumber]] = NSNumber(value: b)
                }
            }
        }
        CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)

        return array
    }
}

extension UIImage {
    convenience init?(pixelBuffer: CVPixelBuffer) {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
            self.init(cgImage: cgImage)
        } else {
            return nil
        }
    }
}
